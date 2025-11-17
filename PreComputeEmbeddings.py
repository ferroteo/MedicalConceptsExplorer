"""
Pre-compute embeddings for biomedical concept codes and save them to disk.
"""

from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
import warnings
import multiprocessing as mp
import os
import argparse

warnings.filterwarnings('ignore')

CONFIG_FILE = './models/config.json'
CONCEPTS_FILE_PATH = "./concept_bridge.parquet"
OUTPUT_DIR = Path('./embeddings')

cpu_cores = os.cpu_count()
torch.set_num_threads(cpu_cores)
torch.set_num_interop_threads(cpu_cores)
os.environ["OMP_NUM_THREADS"] = str(cpu_cores)
os.environ["MKL_NUM_THREADS"] = str(cpu_cores)


def load_config():
    """Load model configuration from JSON file"""
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
        return config.get('MODELS_CONFIG', {})
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file '{CONFIG_FILE}' not found")
    except json.JSONDecodeError:
        raise ValueError(f"Configuration file '{CONFIG_FILE}' is not valid JSON")


def check_embeddings_exist(model_name, vocabulary):
    """Check if embeddings already exist for this model and vocabulary"""
    model_dir = OUTPUT_DIR / model_name / vocabulary
    embeddings_file = model_dir / 'embeddings.npy'
    return embeddings_file.exists()


def load_concepts(filepath, vocabulary='OMOP'):
    """Load concepts filtered by vocabulary and remove duplicates"""
    print(f"Loading concepts from {filepath}...")
    print(f"Filtering for vocabulary: {vocabulary}")
    
    concepts_df = pd.read_parquet(filepath)
    
    required_cols = {'vocabulary', 'code', 'name'}
    if not required_cols.issubset(concepts_df.columns):
        raise ValueError(f"Expected columns {required_cols} not found in {filepath}")
    
    concepts_df = concepts_df[concepts_df['vocabulary'] == vocabulary]
    
    print(f"Found {len(concepts_df)} concepts before deduplication")
    concepts_df = concepts_df.drop_duplicates(subset=['vocabulary', 'code', 'name'])
    print(f"After removing duplicates: {len(concepts_df)} concepts")
    
    codes = concepts_df['code'].tolist()
    # names = concepts_df['name'].tolist()
    
    # combined_input = [f"[{vocab}: {code}, {name}]" for vocab, code, name in zip(concepts_df['vocabulary'], codes, names)]
    # text_input = names
    text_input = codes
    
    print(f"Loaded {len(text_input)} unique concepts from vocabulary '{vocabulary}'")
    return text_input


def _process_transformer_chunk(args):
    """Process a chunk of texts in a separate process"""
    texts, model_name, tokenizer_name, batch_size, chunk_idx = args
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    torch.set_num_threads(max(1, cpu_cores // 4))
    
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            output = model(**encoded)
        
        attention_mask = encoded['attention_mask'].unsqueeze(-1)
        masked_embeddings = output.last_hidden_state * attention_mask
        summed = masked_embeddings.sum(dim=1)
        counts = attention_mask.sum(dim=1)
        mean_pooled = summed / counts
        
        all_embeddings.append(mean_pooled.numpy())
    
    result = np.vstack(all_embeddings)
    print(f"  Chunk {chunk_idx + 1} completed: {len(texts)} texts processed")
    return result


def get_transformer_embeddings_parallel(texts, model_name, tokenizer_name, batch_size=64, num_workers=4):
    """Generate embeddings using parallel processing"""
    print(f"Using CPU with {cpu_cores} cores")
    print(f"PyTorch using {torch.get_num_threads()} threads")
    print(f"Using {num_workers} parallel workers with batch size {batch_size}")
    
    chunk_size = (len(texts) + num_workers - 1) // num_workers
    text_chunks = [texts[i:i+chunk_size] for i in range(0, len(texts), chunk_size)]
    
    worker_args = [
        (chunk, model_name, tokenizer_name, batch_size, idx)
        for idx, chunk in enumerate(text_chunks)
    ]
    
    with mp.Pool(num_workers) as pool:
        results = pool.map(_process_transformer_chunk, worker_args)
    
    return np.vstack(results)


def get_transformer_embeddings(texts, model, tokenizer, batch_size=64):
    """Generate embeddings using transformer models with larger batch size"""
    all_embeddings = []
    model.eval()

    for i in tqdm(range(0, len(texts), batch_size), desc='Embedding Texts'):
        batch_texts = texts[i:i+batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        with torch.no_grad():
            output = model(**encoded)

        attention_mask = encoded['attention_mask'].unsqueeze(-1)
        masked_embeddings = output.last_hidden_state * attention_mask
        summed = masked_embeddings.sum(dim=1)
        counts = attention_mask.sum(dim=1)
        mean_pooled = summed / counts

        all_embeddings.append(mean_pooled.numpy())

    return np.vstack(all_embeddings)


def get_sentence_transformer_embeddings(texts, model, batch_size=64):
    """Generate embeddings using sentence transformer models"""
    return model.encode(
        texts, 
        batch_size=batch_size,  
        show_progress_bar=True, 
        convert_to_numpy=True
    )


def precompute_embeddings_for_model(model_name, text_input, config, use_parallel=True):
    """Pre-compute embeddings for a specific model"""
    print(f"\n{'='*80}")
    print(f"Processing model: {model_name.upper()}")
    print(f"{'='*80}")
    
    model_config = config[model_name]
    
    if model_config['type'] == 'transformer':
        print(f"Loading {model_name}...")
        
        if use_parallel:
            print("Generating embeddings with parallel processing...")
            embeddings = get_transformer_embeddings_parallel(
                text_input,
                model_config['model'],
                model_config['tokenizer'],
                batch_size=256,
                num_workers=4
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_config['tokenizer'])
            model = AutoModel.from_pretrained(model_config['model'])
            
            print("Generating embeddings (single process)...")
            embeddings = get_transformer_embeddings(
                text_input, model, tokenizer, batch_size=64
            )
        
    elif model_config['type'] == 'sentence_transformer':
        print(f"Loading {model_name}...")
        model = SentenceTransformer(model_config['path'])
        
        print("Generating embeddings...")
        embeddings = get_sentence_transformer_embeddings(
            text_input, model, batch_size=64
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_config['type']}")
    
    return embeddings


def save_embeddings(model_name, embeddings, vocabulary):
    """Save embeddings as numpy array"""
    model_dir = OUTPUT_DIR / model_name / vocabulary
    model_dir.mkdir(parents=True, exist_ok=True)
    
    embeddings_path = model_dir / 'embeddings.npy'
    np.save(embeddings_path, embeddings)
    
    print(f"Saved embeddings: shape {embeddings.shape}")
    print(f"Location: {embeddings_path}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Pre-compute embeddings for biomedical concept codes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python PreComputeEmbeddings.py
  python PreComputeEmbeddings.py --model bridge --vocabulary ICD10CM
  python PreComputeEmbeddings.py --model all --vocabulary OMOP --parallel
  python PreComputeEmbeddings.py --model bridge gatortron --vocabulary ATC --no-parallel
        """
    )
    
    parser.add_argument(
        '--model',
        nargs='+',
        default=['gatortron'],
        help='Model(s) to use for embedding generation. Use "all" for all models, or specify model names (default: gatortron)'
    )
    
    parser.add_argument(
        '--vocabulary',
        type=str,
        default='OMOP',
        help='Vocabulary to filter concepts (default: OMOP)'
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        default=False,
        help='Use parallel processing for transformer models (default: False)'
    )
    
    parser.add_argument(
        '--no-parallel',
        dest='parallel',
        action='store_false',
        help='Disable parallel processing (use single process)'
    )
    
    return parser.parse_args()


def main():
    """Main pre-computation pipeline"""
    args = parse_arguments()
    
    print("Biomedical Concept Embedding Pre-computation Pipeline")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Vocabulary: {args.vocabulary}")
    print(f"  - Parallel Processing: {args.parallel}")
    print(f"  - Requested Models: {args.model}")
    print("="*80)
    
    MODELS_CONFIG = load_config()
    
    if 'all' in args.model:
        models_to_process = list(MODELS_CONFIG.keys())
        print(f"Processing all available models: {models_to_process}")
    else:
        models_to_process = args.model
        invalid_models = [m for m in models_to_process if m not in MODELS_CONFIG]
        if invalid_models:
            print(f"Warning: Unknown models will be skipped: {invalid_models}")
        models_to_process = [m for m in models_to_process if m in MODELS_CONFIG]
        
        if not models_to_process:
            print("Error: No valid models specified")
            print(f"Available models: {list(MODELS_CONFIG.keys())}")
            return
    
    print("\nChecking for existing embeddings...")
    models_to_skip = []
    models_to_compute = []
    
    for model_name in models_to_process:
        if check_embeddings_exist(model_name, args.vocabulary):
            models_to_skip.append(model_name)
            print(f"  ✓ {model_name}: Embeddings already exist (skipping)")
        else:
            models_to_compute.append(model_name)
            print(f"  • {model_name}: Will compute embeddings")
    
    if not models_to_compute:
        print("\n" + "="*80)
        print("All requested embeddings already exist. Nothing to compute.")
        print(f"Embeddings location: {OUTPUT_DIR.absolute()}")
        print("="*80)
        return
    
    print(f"\nModels to compute: {len(models_to_compute)}/{len(models_to_process)}")
    
    text_input = load_concepts(CONCEPTS_FILE_PATH, args.vocabulary)
    
    if len(text_input) == 0:
        print(f"Error: No concepts found for vocabulary '{args.vocabulary}'")
        return
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    import time
    for model_name in models_to_compute:
        try:
            start_time = time.time()
            embeddings = precompute_embeddings_for_model(
                model_name, 
                text_input,
                MODELS_CONFIG,
                use_parallel=args.parallel
            )
            save_embeddings(model_name, embeddings, args.vocabulary)
            elapsed = time.time() - start_time
            print(f"✓ Successfully processed {model_name} in {elapsed:.2f} seconds")
        except Exception as e:
            print(f"✗ Error processing {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*80)
    print("Pre-computation complete!")
    print(f"Embeddings saved to: {OUTPUT_DIR.absolute()}")
    if models_to_skip:
        print(f"Skipped {len(models_to_skip)} model(s) with existing embeddings: {models_to_skip}")
    print("="*80)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()