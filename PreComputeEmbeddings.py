"""
Pre-compute embeddings for ICD-10 codes and save them to disk.
"""

from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import numpy as np
import torch
import pickle
import json
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Configuration
MODELS_CONFIG = {
    'gatortron': {
        'tokenizer': 'UFNLP/gatortron-base',
        'model': 'UFNLP/gatortron-base',
        'type': 'transformer'
    },
    'bridge': {
        'path': '/home/mattferr/Projects/EmbeddingComparisons/Embedding-SBERT-CLIP-256-Full-woGraph-woCandidate-STRONG/sbert/',
        'type': 'sentence_transformer'
    }
}

OUTPUT_DIR = Path('precomputed_embeddings')
ICD10_FILE_PATH = "/home/mattferr/Projects/EmbeddingComparisons/icd10_codes.csv"

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def load_icd10_codes(filepath):
    """Load all ICD-10-CM codes and descriptions"""
    print(f"Loading ICD-10 codes from {filepath}...")
    icd_df = pd.read_csv(filepath)

    code_col = 'Code'
    desc_col = 'LongDescription'

    if code_col not in icd_df.columns or desc_col not in icd_df.columns:
        raise ValueError(f"Expected columns '{code_col}' and '{desc_col}' not found.")

    codes = icd_df[code_col].tolist()
    descriptions = icd_df[desc_col].tolist()
    # Combine medical vocabulary, code and description in one unique string
    combined_input = [f"[ICD-10-CM: {code}, {desc}]" for code, desc in zip(codes, descriptions)]

    print(f"Loaded {len(codes)} ICD-10-CM codes")
    return combined_input


def get_transformer_embeddings(texts, model, tokenizer, batch_size=8):
    """Generate embeddings using transformer models"""
    all_embeddings = []
    model.eval()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output = model(**encoded)

        # Mean pooling
        attention_mask = encoded['attention_mask'].unsqueeze(-1)
        masked_embeddings = output.last_hidden_state * attention_mask
        summed = masked_embeddings.sum(dim=1)
        counts = attention_mask.sum(dim=1)
        mean_pooled = summed / counts

        all_embeddings.append(mean_pooled.cpu().numpy())

        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {i + len(batch_texts)}/{len(texts)} texts")

    return np.vstack(all_embeddings)


def precompute_embeddings_for_model(model_name, combined_input):
    """Pre-compute embeddings for a specific model"""
    print(f"\n{'='*80}")
    print(f"Processing model: {model_name.upper()}")
    print(f"{'='*80}")
    
    config = MODELS_CONFIG[model_name]
    
    # Load model
    if config['type'] == 'transformer':
        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])
        model = AutoModel.from_pretrained(config['model']).to(device)
        
        print("Generating embeddings for combined text...")
        combined_input_embeddings = get_transformer_embeddings(combined_input, model, tokenizer)
        
    elif config['type'] == 'sentence_transformer':
        print(f"Loading {model_name}...")
        model = SentenceTransformer(config['path'])
        
        print("Generating embeddings for combined text...")
        combined_input_embeddings = model.encode(combined_input, show_progress_bar=True, convert_to_numpy=True)
    
    return combined_input_embeddings


def save_embeddings(model_name, combined_input_embeddings):
    """Save embeddings and metadata to disk"""
    model_dir = OUTPUT_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save embeddings as parquet
    df = pd.DataFrame(combined_input_embeddings)
    df.to_parquet(model_dir / f'embeddings.parquet', index=False)
    print(f"Saved embeddings: shape {combined_input_embeddings.shape}")


def main():
    """Main pre-computation pipeline"""
    print("ICD-10-CM Embedding Pre-computation Pipeline")
    print("="*80)
    
    # Load ICD-10 codes
    combined_input = load_icd10_codes(ICD10_FILE_PATH)
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Process each model
    for model_name in MODELS_CONFIG.keys():
        try:
            embeddings = precompute_embeddings_for_model(model_name, combined_input)
            save_embeddings(model_name, embeddings)
            print(f"✓ Successfully processed {model_name}")
        except Exception as e:
            print(f"✗ Error processing {model_name}: {str(e)}")
            continue
    
    print("\n" + "="*80)
    print("Pre-computation complete!")
    print(f"Embeddings saved to: {OUTPUT_DIR.absolute()}")
    print("="*80)


if __name__ == '__main__':
    main()