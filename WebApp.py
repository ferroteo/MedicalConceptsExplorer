"""
Simple Streamlit app to find similar ICD-10-CM concept (combined code & description) based on embeddings
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

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

EMBEDDINGS_DIR = Path('precomputed_embeddings')
ICD10_FILE_PATH = "/home/mattferr/Projects/EmbeddingComparisons/icd10_codes.csv"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@st.cache_data
def load_icd10_codes(icd_chars=3):
    icd_df = pd.read_csv(ICD10_FILE_PATH)
    # Filter by exact code length
    icd_df = icd_df[icd_df['Code'].str.len() == icd_chars]
    # Combine medical vocabulary, code and description in one unique string
    icd_df['combined_input'] = icd_df.apply(lambda x: f"[ICD-10-CM: {x['Code']}, {x['LongDescription']}]", axis=1)
    return icd_df


@st.cache_data
def load_embeddings(model_name, icd_chars):
    """Load precomputed embeddings for selected model and ICD length"""
    embedding_file = EMBEDDINGS_DIR / model_name / f"icd{icd_chars}" / 'embeddings.parquet'
    if not embedding_file.exists():
        raise FileNotFoundError(f"Embeddings not found for model '{model_name}' and ICD length {icd_chars}. Expected at {embedding_file}")
    embeddings = pd.read_parquet(embedding_file).values
    return embeddings


@st.cache_resource
def load_model(model_name):
    """Load model if available"""
    if model_name not in MODELS_CONFIG:
        raise ValueError(f"Model '{model_name}' not found in configuration.")
    config = MODELS_CONFIG[model_name]

    if config['type'] == 'transformer':
        tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])
        model = AutoModel.from_pretrained(config['model']).to(device)
        model.eval()
        return {'tokenizer': tokenizer, 'model': model, 'type': 'transformer'}
    elif config['type'] == 'sentence_transformer':
        model = SentenceTransformer(config['path'])
        return {'model': model, 'type': 'sentence_transformer'}
    else:
        raise ValueError(f"Unknown model type for '{model_name}'.")


def get_user_embedding(text, model_dict):
    """Generate embedding for user input"""
    if model_dict['type'] == 'transformer':
        tokenizer = model_dict['tokenizer']
        model = model_dict['model']
        encoded = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**encoded)
        attention_mask = encoded['attention_mask'].unsqueeze(-1)
        masked_embeddings = output.last_hidden_state * attention_mask
        summed = masked_embeddings.sum(dim=1)
        counts = attention_mask.sum(dim=1)
        mean_pooled = summed / counts
        return mean_pooled.cpu().numpy()
    elif model_dict['type'] == 'sentence_transformer':
        model = model_dict['model']
        return model.encode([text], convert_to_numpy=True)


def find_similar_codes(user_text, model_name, icd_data, icd_embeddings, top_n=10):
    """Compute cosine similarities and return ranked ICD codes"""
    model_dict = load_model(model_name)
    user_embedding = get_user_embedding(user_text, model_dict)
    similarities = cosine_similarity(user_embedding, icd_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_n]
    results = pd.DataFrame({
        'Rank': range(1, len(top_indices) + 1),
        'CombinedCodeDescription': icd_data.iloc[top_indices]['combined_input'].values,
        'Similarity': similarities[top_indices]
    })
    return results


def main():
    st.set_page_config(page_title="ICD-10-CM Similarity Finder", layout="wide")
    st.title("🏥 ICD-10-CM Code Finder")
    st.markdown("Find the most similar ICD-10-CM codes based on clinical text")

    # Sidebar
    st.sidebar.header("Settings")
    model_name = st.sidebar.selectbox("Select Model", options=list(MODELS_CONFIG.keys()), index=0)  # default gatortron
    icd_chars = st.sidebar.number_input("ICD Code Length", min_value=3, max_value=7, value=3, step=1)
    top_n = st.sidebar.slider("Number of Results", min_value=1, max_value=50, value=10)

    # Load data
    with st.spinner("Loading ICD-10 codes and embeddings..."):
        try:
            icd_data = load_icd10_codes()
            icd_embeddings = load_embeddings(model_name, icd_chars)
        except Exception as e:
            st.error(f"❌ Error loading data or embeddings: {str(e)}")
            return

    st.sidebar.success(f"✓ Loaded {len(icd_data)} ICD-10 codes")
    st.sidebar.info(f"Model: {model_name}\nICD length: {icd_chars}\nDevice: {device}")

    # User input
    user_input = st.text_area(
        "Enter Clinical Text:",
        placeholder="e.g., Patient has acute bronchitis with persistent cough and fever",
        height=100
    )

    st.info("💡 The app compares your text against precomputed embeddings of combined ICD-10-CM codes and descriptions.")

    search_button = st.button("🔍 Find Similar Codes", type="primary")

    if search_button and user_input.strip():
        try:
            with st.spinner("Computing similarities..."):
                results = find_similar_codes(user_input, model_name, icd_data, icd_embeddings, top_n)
        except Exception as e:
            st.error(f"❌ Error during similarity computation: {str(e)}")
            return

        st.success(f"Found {len(results)} similar codes!")

        st.markdown("### 🎯 Top Match")
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            st.metric("Rank", int(results.iloc[0]['Rank']))
        with col2:
            st.write(results.iloc[0]['CombinedCodeDescription'])
        with col3:
            st.metric("Similarity", f"{results.iloc[0]['Similarity']:.4f}")

        st.markdown("### 📊 All Results")
        display_results = results[['Rank', 'CombinedCodeDescription', 'Similarity']].copy()

        def color_similarity(val):
            color = f'background-color: rgba(255, {int(255 * (1 - val))}, {int(255 * (1 - val))}, 0.5)'
            return color

        styled_results = display_results.style.applymap(color_similarity, subset=['Similarity']).format({'Similarity': '{:.4f}'})
        st.dataframe(styled_results, use_container_width=True, height=400)

        csv = results.to_csv(index=False)
        st.download_button("📥 Download Results as CSV", csv, "icd10_similarity_results.csv", "text/csv")

    elif search_button and not user_input.strip():
        st.warning("⚠️ Please enter some clinical text to search")


if __name__ == '__main__':
    main()
