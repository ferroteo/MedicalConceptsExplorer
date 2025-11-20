"""
Enhanced Streamlit app for biomedical ontology retrieval on Hugging Face
Supports multiple vocabularies with precomputed embeddings and t-SNE visualization
Now supports multiple embedding models: BRIDGE, SapBERT, and GatorTron
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import altair as alt
import toml

# Model configuration
MODELS_CONFIG = {
    "bridge": {
        "display_name": "BRIDGE",
        "type": "sentence_transformer",
        "repo": "dsgelab/BRIDGE"
    },
    "sapbert": {
        "display_name": "SapBERT",
        "type": "transformer",
        "tokenizer": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "model": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    }
}

# Get HF token from environment (for HF Spaces) or secrets (for local/Streamlit Cloud)
def get_hf_token():
    """Get HF token from environment or secrets"""
    token = os.getenv("HF_TOKEN")
    if not token:
        try:
            # Try multiple potential secrets.toml locations
            for secrets_path in ["/root/.streamlit/secrets.toml", "/app/.streamlit/secrets.toml", "/app/src/.streamlit/secrets.toml"]:
                if os.path.exists(secrets_path):
                    secrets = toml.load(secrets_path)
                    token = secrets.get("HF_TOKEN")
                    if token:
                        break
            # Fallback to Streamlit's secrets if available
            if not token:
                token = st.secrets.get("HF_TOKEN")
        except:
            pass
    return token

# Configuration
HF_USERNAME = "dsgelab"
DATASET_REPO = f"{HF_USERNAME}/BRIDGE_data"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cache functions
@st.cache_data
def load_concepts():
    """Load biomedical concepts from Hugging Face dataset"""
    try:
        hf_token = get_hf_token()
        
        # Download the parquet file from Hugging Face
        file_path = hf_hub_download(
            repo_id=DATASET_REPO,
            filename="concepts_bridge_appsubset.parquet",
            repo_type="dataset",
            token=hf_token
        )
        df = pd.read_parquet(file_path)
        
        # Verify required columns
        required_cols = {"vocabulary", "code", "name"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Missing required columns: {required_cols - set(df.columns)}")
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['vocabulary', 'code', 'name']).reset_index(drop=True)
        
        return df
    except Exception as e:
        st.error(f"Error loading concepts: {e}")
        raise

@st.cache_data
def load_embeddings(vocabulary, model_name):
    """Load precomputed embeddings for a specific vocabulary and model from Hugging Face dataset"""
    try:
        hf_token = get_hf_token()
        
        # Download the embeddings file for this vocabulary and model
        file_path = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=f"embeddings/{model_name}/{vocabulary}/embeddings.npy",
            repo_type="dataset",
            token=hf_token
        )
        embeddings = np.load(file_path)
        return embeddings
    except Exception as e:
        st.error(f"Error loading embeddings for {vocabulary} with {model_name}: {e}")
        raise

@st.cache_resource
def load_sentence_transformer_model(model_name):
    """Load a SentenceTransformer model"""
    hf_token = get_hf_token()
    config = MODELS_CONFIG[model_name]
    
    try:
        model = SentenceTransformer(
            config["repo"],
            token=hf_token,
            device=device
        )
        return model
    except Exception as e:
        st.error(f"Error loading {config['display_name']} model: {e}")
        raise

@st.cache_resource
def load_transformer_model(model_name):
    """Load a Hugging Face Transformer model"""
    hf_token = get_hf_token()
    config = MODELS_CONFIG[model_name]
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config["tokenizer"],
            token=hf_token
        )
        model = AutoModel.from_pretrained(
            config["model"],
            token=hf_token
        ).to(device)
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading {config['display_name']} model: {e}")
        raise

def mean_pooling(model_output, attention_mask):
    """Mean pooling for transformer models"""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_user_embedding_transformer(text, tokenizer, model):
    """Generate embedding for user input text using transformer model"""
    # Tokenize
    encoded_input = tokenizer(
        [text],
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=512
    ).to(device)
    
    # Generate embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Mean pooling
    embedding = mean_pooling(model_output, encoded_input['attention_mask'])
    
    # Normalize
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
    
    return embedding.cpu().numpy()

def get_user_embedding_sentence_transformer(text, model):
    """Generate embedding for user input text using SentenceTransformer"""
    embedding = model.encode([text], convert_to_numpy=True, device=device)
    return embedding

def get_user_embedding(text, model_name):
    """Generate embedding for user input text based on model type"""
    config = MODELS_CONFIG[model_name]
    
    if config["type"] == "sentence_transformer":
        model = load_sentence_transformer_model(model_name)
        return get_user_embedding_sentence_transformer(text, model)
    else:  # transformer
        tokenizer, model = load_transformer_model(model_name)
        return get_user_embedding_transformer(text, tokenizer, model)

def find_similar_concepts(user_text, df_concepts, embeddings_dict, model_name, top_n=30):
    """Find most similar biomedical concepts across vocabularies"""
    user_embedding = get_user_embedding(user_text, model_name)
    
    # Calculate similarities for each vocabulary
    all_results = []
    
    for vocab, emb_array in embeddings_dict.items():
        vocab_mask = df_concepts["vocabulary"] == vocab
        vocab_df = df_concepts[vocab_mask].reset_index(drop=True)
        
        # Verify alignment
        if len(vocab_df) != len(emb_array):
            st.warning(f"Skipping {vocab}: concept count mismatch ({len(vocab_df)} vs {len(emb_array)})")
            continue
        
        # Calculate cosine similarities
        similarities = cosine_similarity(user_embedding, emb_array)[0]
        
        # Create results for this vocabulary
        vocab_results = pd.DataFrame({
            'vocabulary': vocab_df['vocabulary'].values,
            'code': vocab_df['code'].values,
            'name': vocab_df['name'].values,
            'similarity': similarities
        })
        
        all_results.append(vocab_results)
    
    # Combine all results
    if not all_results:
        return pd.DataFrame()
    
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Sort by similarity and get top N
    combined_results = combined_results.sort_values('similarity', ascending=False).head(top_n)
    combined_results['rank'] = range(1, len(combined_results) + 1)
    
    return combined_results

def compute_tsne_plot(df_concepts, embeddings_dict, query_embedding, selected_vocabs, max_points=5000):
    """Create t-SNE visualization of concepts and query"""
    
    # Collect embeddings and metadata for selected vocabularies
    emb_list = []
    vocab_list = []
    code_list = []
    name_list = []
    
    for vocab in selected_vocabs:
        if vocab not in embeddings_dict:
            continue
            
        vocab_mask = df_concepts["vocabulary"] == vocab
        vocab_df = df_concepts[vocab_mask].reset_index(drop=True)
        emb_array = embeddings_dict[vocab]
        
        if len(vocab_df) != len(emb_array):
            continue
        
        emb_list.append(emb_array)
        vocab_list.extend(vocab_df['vocabulary'].tolist())
        code_list.extend(vocab_df['code'].tolist())
        name_list.extend(vocab_df['name'].tolist())
    
    if not emb_list:
        return None
    
    # Stack all embeddings
    all_emb = np.vstack(emb_list)
    
    # Sample if too many points
    n_points = all_emb.shape[0]
    if n_points > max_points:
        idx = np.random.choice(n_points, max_points, replace=False)
        all_emb = all_emb[idx]
        vocab_list = [vocab_list[i] for i in idx]
        code_list = [code_list[i] for i in idx]
        name_list = [name_list[i] for i in idx]
    
    # Add query embedding
    all_emb_with_query = np.vstack([all_emb, query_embedding])
    
    # Compute t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="random",
        random_state=42,
    )
    coords = tsne.fit_transform(all_emb_with_query)
    
    # Create visualization dataframe
    df_concepts_viz = pd.DataFrame({
        "x": coords[:-1, 0],
        "y": coords[:-1, 1],
        "type": "concept",
        "vocabulary": vocab_list,
        "code": code_list,
        "name": name_list,
    })
    
    df_query = pd.DataFrame({
        "x": [coords[-1, 0]],
        "y": [coords[-1, 1]],
        "type": ["query"],
        "vocabulary": [""],
        "code": [""],
        "name": ["Query"],
    })
    
    df_vis = pd.concat([df_concepts_viz, df_query], ignore_index=True)
    
    # Create Altair chart
    chart = (
        alt.Chart(df_vis)
        .mark_circle()
        .encode(
            x=alt.X("x:Q", title="t-SNE Dimension 1"),
            y=alt.Y("y:Q", title="t-SNE Dimension 2"),
            color=alt.Color(
                "type:N",
                scale=alt.Scale(domain=["concept", "query"], range=["steelblue", "red"]),
                legend=alt.Legend(title="Type"),
            ),
            tooltip=[
                alt.Tooltip("vocabulary:N", title="Vocabulary"),
                alt.Tooltip("code:N", title="Code"),
                alt.Tooltip("name:N", title="Name"),
            ],
            size=alt.condition("datum.type == 'query'", alt.value(200), alt.value(60)),
        )
        .properties(width=800, height=600)
        .interactive()
    )
    
    return chart

# Streamlit App
def main():
    st.set_page_config(page_title="Biomedical Ontology Retrieval", layout="wide")
    
    st.title("🧬 Biomedical Ontology Retrieval")
    st.markdown("Find similar concepts across multiple biomedical vocabularies using state-of-the-art embeddings")
    
    # Check for HF token
    if not get_hf_token():
        st.warning("⚠️ No Hugging Face token found. If your repos are private, add HF_TOKEN as a repository secret.")
    
    # Load data
    with st.spinner("Loading concepts and embeddings from Hugging Face..."):
        try:
            df_concepts = load_concepts()
            available_vocabs = sorted(df_concepts["vocabulary"].unique().tolist())
        except Exception as e:
            st.error("Failed to load concept data. Please check your Hugging Face configuration.")
            st.stop()
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Settings")
    
    # Model selection
    model_options = {name: config["display_name"] for name, config in MODELS_CONFIG.items()}
    selected_model_key = st.sidebar.selectbox(
        "Embedding Model",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=0,
        help="Select which embedding model to use for encoding your query"
    )
    
    selected_model_name = MODELS_CONFIG[selected_model_key]["display_name"]
    
    # Vocabulary selection
    selected_vocabs = st.sidebar.multiselect(
        "Vocabularies",
        available_vocabs,
        default=available_vocabs,
        help="Select which biomedical vocabularies to search"
    )
    
    if not selected_vocabs:
        st.error("Please select at least one vocabulary.")
        st.stop()
    
    # Load embeddings for selected vocabularies and model
    with st.spinner(f"Loading {selected_model_name} embeddings for selected vocabularies..."):
        embeddings_dict = {}
        failed_vocabs = []
        
        for vocab in selected_vocabs:
            try:
                embeddings_dict[vocab] = load_embeddings(vocab, selected_model_key)
            except Exception as e:
                failed_vocabs.append(vocab)
                st.warning(f"Could not load {selected_model_name} embeddings for {vocab}")
        
        # Update selected vocabs to only those with successful loading
        selected_vocabs = [v for v in selected_vocabs if v in embeddings_dict]
        
        if not selected_vocabs:
            st.error(f"No vocabularies could be loaded successfully with {selected_model_name}.")
            st.stop()
    
    # Number of results slider
    top_n = st.sidebar.slider(
        "Number of Results",
        min_value=5,
        max_value=100,
        value=30,
        step=5
    )
    
    # Show t-SNE visualization
    show_tsne = st.sidebar.checkbox(
        "Show t-SNE Visualization",
        value=True,
        help="Display 2D projection of concept embeddings (may take a moment to compute)"
    )
    
    # Info in sidebar
    st.sidebar.success(f"✓ Loaded {len(df_concepts)} concepts")
    st.sidebar.info(f"📊 Active vocabularies: {len(selected_vocabs)}")
    st.sidebar.info(f"🤖 Model: {selected_model_name}\n💻 Device: {device}")
    
    # Display vocabulary statistics
    with st.sidebar.expander("📈 Vocabulary Statistics"):
        for vocab in selected_vocabs:
            vocab_count = (df_concepts["vocabulary"] == vocab).sum()
            st.write(f"**{vocab}**: {vocab_count:,} concepts")
    
    # Main area
    st.subheader("🔍 Query")
    user_input = st.text_area(
        "Enter clinical or biomedical text:",
        placeholder="e.g., Migraine with aura",
        height=120,
        help="Enter any clinical description or biomedical concept to find similar codes"
    )
    
    st.info(f"💡 The system searches across all selected vocabularies using precomputed {selected_model_name} embeddings for fast retrieval.")
    
    search_button = st.button("🔎 Search Similar Concepts", type="primary", use_container_width=True)
    
    if search_button and user_input.strip():
        with st.spinner("Finding similar concepts..."):
            try:
                results = find_similar_concepts(
                    user_input,
                    df_concepts,
                    embeddings_dict,
                    selected_model_key,
                    top_n
                )
                
                if results.empty:
                    st.warning("No results found. Please try a different query.")
                    st.stop()
                
            except Exception as e:
                st.error(f"Error during search: {e}")
                st.stop()
        
        # Display results
        st.success(f"✓ Found {len(results)} similar concepts!")
        
        # Results table
        st.markdown("### 📊 All Results")
        
        display_results = results[['rank', 'vocabulary', 'code', 'name', 'similarity']].copy()
        display_results.columns = ['Rank', 'Vocabulary', 'Code', 'Name', 'Similarity']
        
        # Style the dataframe
        def color_similarity(val):
            if isinstance(val, (int, float)):
                intensity = int(255 * (1 - val))
                return f'background-color: rgba(255, {intensity}, {intensity}, 0.5)'
            return ''
        
        styled_results = display_results.style.applymap(
            color_similarity,
            subset=['Similarity']
        ).format({'Similarity': '{:.4f}'})
        
        st.dataframe(styled_results, use_container_width=True, height=400)
        
        # t-SNE visualization
        if show_tsne:
            st.markdown("### 🗺️ t-SNE Visualization")
            st.info("This shows a 2D projection of the concept embeddings. Your query is shown in red.")
            
            with st.spinner("Computing t-SNE projection..."):
                try:
                    query_embedding = get_user_embedding(user_input, selected_model_key)
                    
                    chart = compute_tsne_plot(
                        df_concepts,
                        embeddings_dict,
                        query_embedding,
                        selected_vocabs,
                        max_points=5000
                    )
                    
                    if chart:
                        st.altair_chart(chart, use_container_width=True)
                    else:
                        st.warning("Could not generate t-SNE visualization.")
                except Exception as e:
                    st.error(f"Error generating visualization: {e}")
        
        # Download button
        csv = results.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"biomedical_ontology_retrieval_results_{selected_model_key}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    elif search_button and not user_input.strip():
        st.warning("⚠️ Please enter some text to search")
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"""
        <div style='text-align: center; color: gray;'>
        <small>Powered by {selected_model_name} embeddings | Built with Streamlit</small>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()