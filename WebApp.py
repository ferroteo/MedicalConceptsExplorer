#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import torch

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import altair as alt


# --------
# ENCODERS
# --------

def _mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class SapBERT:
    def __init__(self, path, device='cuda'):
        self.tokenizer = AutoTokenizer.from_pretrained(path, cache_dir="Models")
        self.transformers_model = AutoModel.from_pretrained(path, cache_dir="Models").to(device)
        self.device = device

    def encode(self, texts, device=None):
        device = device or self.device
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False

        bs = 128
        all_embs = []
        for i in np.arange(0, len(texts), bs):
            toks = self.tokenizer.batch_encode_plus(
                texts[i:i+bs],
                padding="max_length",
                max_length=25,
                truncation=True,
                return_tensors="pt"
            )
            toks_cuda = {k: v.to(device) for k, v in toks.items()}
            cls_rep = self.transformers_model(**toks_cuda)[0][:, 0, :]
            all_embs.append(cls_rep.cpu().detach().numpy())

        final_embeddings = np.concatenate(all_embs, axis=0)
        if single_input:
            return final_embeddings[0]
        return final_embeddings


class BioBERT:
    def __init__(self, path: str, device: str = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(path, cache_dir="Models")
        self.transformers_model = AutoModel.from_pretrained(path, cache_dir="Models").to(device)
        self.device = device

    def encode(self, texts, max_length: int = 256):
        if isinstance(texts, str):
            texts = [texts]

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            out = self.transformers_model(**enc)

        sent = _mean_pooling(out.last_hidden_state, enc["attention_mask"])
        sent = torch.nn.functional.normalize(sent, p=2, dim=1)
        return sent.cpu().numpy()


class ClinicalBERT:
    def __init__(self, path: str, device: str = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(path, cache_dir="Models")
        self.transformers_model = AutoModel.from_pretrained(path, cache_dir="Models").to(device)
        self.device = device

    def encode(self, texts, max_length: int = 256):
        if isinstance(texts, str):
            texts = [texts]

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            out = self.transformers_model(**enc)

        sent = _mean_pooling(out.last_hidden_state, enc["attention_mask"])
        sent = torch.nn.functional.normalize(sent, p=2, dim=1)
        return sent.cpu().numpy()


class Gatortron:
    def __init__(self, path: str, device: str = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(path, cache_dir="Models")
        self.transformers_model = AutoModel.from_pretrained(path, cache_dir="Models").to(device)
        self.device = device

    def encode(self, texts, max_length: int = 256):
        if isinstance(texts, str):
            texts = [texts]

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            out = self.transformers_model(**enc)

        sent = _mean_pooling(out.last_hidden_state, enc["attention_mask"])
        sent = torch.nn.functional.normalize(sent, p=2, dim=1)
        return sent.cpu().numpy()


class BioMistral:
    def __init__(self, path, device="cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(path, cache_dir="Models")
        # ensure we have a pad token
        if self.tokenizer.pad_token is None:
            # many Mistral/LLaMA-style tokenizers use eos as pad
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.transformers_model = AutoModel.from_pretrained(path, cache_dir="Models").to(device)

    def encode(self, texts, device=None):
        device = device or self.device
        if isinstance(texts, str):
            texts = [texts]

        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = self.transformers_model(**encoded_input)

        token_embeddings = outputs.last_hidden_state          # (B, L, D)
        attention_mask = encoded_input["attention_mask"]      # (B, L)

        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)   # (B, D)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)            # (B, 1)
        mean_embeddings = sum_embeddings / sum_mask                           # (B, D)

        return mean_embeddings.cpu().numpy()


# -----------
# APP CONFIG
# -----------

DATA_PARQUET = Path("concepts_bridge_processed.parquet")
EMB_DIR = Path("./embeddings")
MODEL_TO_PREFIX = {
    "BRIDGE": "bridge",
    "SapBERT": "sapbert",
    "BioBERT": "biobert",
    "ClinicalBERT": "clinicalbert",
    "Gatortron-base": "gatortron_base",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_data
def load_df() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PARQUET)
    needed = {"vocabulary", "code", "name", "text_vc", "text_name"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in parquet: {missing}")
    return df


@st.cache_resource
def load_embedding_array(model_label: str, which: str) -> np.ndarray:
    assert which in ("vc", "name")
    prefix = MODEL_TO_PREFIX[model_label]
    path = EMB_DIR / f"{prefix}_{which}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Expected embedding file not found: {path}")
    # memory-mapped, read-only
    arr = np.load(path, mmap_mode="r")
    return arr


@st.cache_resource
def load_query_encoder(model_label: str):
    if model_label == "SapBERT":
        return ("sapbert", SapBERT("cambridgeltl/SapBERT-from-PubMedBERT-fulltext", device=DEVICE))
    if model_label == "BioBERT":
        return ("transformer", BioBERT("dmis-lab/biobert-v1.1", device=DEVICE))
    if model_label == "ClinicalBERT":
        return ("transformer", ClinicalBERT("emilyalsentzer/Bio_ClinicalBERT", device=DEVICE))
    if model_label == "Gatortron-base":
        return ("transformer", Gatortron("UFNLP/gatortron-base", device=DEVICE))
    if model_label == "BioMistral-7B":
        return ("transformer", BioMistral("BioMistral/BioMistral-7B", device=DEVICE))
    if model_label == "BRIDGE":
        return ("sentence_transformer", SentenceTransformer("/scratch/project_2007428/users/burian/Documents/DelphiEmbeddings/models/Embedding-SBERT-CLIP-64-Full-woGraph-woCandidate-OMOP/sbert", device=DEVICE))
    raise ValueError(f"Unknown model label: {model_label}")


def embed_query_with_same_class(query: str, enc_tuple):
    enc_type, enc = enc_tuple
    if enc_type == "sentence_transformer":
        return enc.encode([query], convert_to_numpy=True)
    else:
        return enc.encode(query)


def compute_tsne_plot(emb_sub: np.ndarray, q_emb: np.ndarray, df_sub: pd.DataFrame, max_points: int = 10_000) -> alt.Chart:
    n_points = emb_sub.shape[0]
    if n_points > max_points:
        idx = np.random.choice(n_points, max_points, replace=False)
        emb_small = emb_sub[idx]
        df_small = df_sub.iloc[idx].reset_index(drop=True)
    else:
        emb_small = emb_sub
        df_small = df_sub

    all_emb = np.vstack([emb_small, q_emb])  # last point is query

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="random",
        random_state=42,
    )
    coords = tsne.fit_transform(all_emb)

    df_icd = pd.DataFrame({
        "x": coords[:-1, 0],
        "y": coords[:-1, 1],
        "type": "concept",
        "vocabulary": df_small["vocabulary"].values,
        "code": df_small["code"].values,
        "name": df_small["name"].values,
    })

    df_query = pd.DataFrame({
        "x": [coords[-1, 0]],
        "y": [coords[-1, 1]],
        "type": ["query"],
        "vocabulary": [""],
        "code": [""],
        "name": [""],
    })

    df_vis = pd.concat([df_icd, df_query], ignore_index=True)

    chart = (
        alt.Chart(df_vis)
        .mark_circle()
        .encode(
            x=alt.X("x:Q", title="t-SNE 1"),
            y=alt.Y("y:Q", title="t-SNE 2"),
            color=alt.Color(
                "type:N",
                scale=alt.Scale(domain=["concept", "query"], range=["steelblue", "red"]),
                legend=alt.Legend(title="Type"),
            ),
            tooltip=[
                alt.Tooltip("vocabulary:N", title="Vocab"),
                alt.Tooltip("code:N", title="Code"),
                alt.Tooltip("name:N", title="Name"),
            ],
            size=alt.condition("datum.type == 'query'", alt.value(160), alt.value(50)),
        )
        .properties(width=800, height=600)
        .interactive()
    )
    return chart


def main():
    st.set_page_config(page_title="Biomedical Ontology Retrieval", layout="wide")
    st.title("Biomedical Ontology Retrieval")

    df = load_df()

    model_label = st.sidebar.selectbox("Model", list(MODEL_TO_PREFIX.keys()), index=0)

    retrieval_source = st.sidebar.radio(
        "Retrieve from",
        ["Codes", "Names"],
        index=0,
    )

    all_vocabs = sorted(df["vocabulary"].unique().tolist())
    selected_vocabs = st.sidebar.multiselect("Vocabularies", all_vocabs, default=all_vocabs)
    if not selected_vocabs:
        st.error("Select at least one vocabulary.")
        return

    top_k = st.sidebar.slider("Top-K", 1, 200, 30, 1)

    if retrieval_source == "Codes":
        emb_arr = load_embedding_array(model_label, "vc")
    else:
        emb_arr = load_embedding_array(model_label, "name")

    mask = df["vocabulary"].isin(selected_vocabs)
    df_sub = df[mask].reset_index(drop=True)
    idx = np.where(mask.values)[0]
    emb_sub = emb_arr[idx]

    st.subheader("Query")
    query = st.text_area("Enter text to retrieve similar concepts", "", height=120)

    if st.button("Search"):
        if not query.strip():
            st.warning("Empty query.")
            return

        enc_tuple = load_query_encoder(model_label)
        q_emb = embed_query_with_same_class(query.strip(), enc_tuple)
        if q_emb.ndim == 1:
            q_emb = q_emb[None, :]

        sims = cosine_similarity(q_emb, emb_sub)[0]

        top_idx = np.argsort(sims)[::-1][:top_k]
        res = df_sub.iloc[top_idx].copy()
        res["similarity"] = sims[top_idx]

        view = res[["vocabulary", "code", "name"]].copy()
        view["similarity"] = res["similarity"].round(4)

        st.subheader("Results")
        st.dataframe(view, use_container_width=True)

        csv = view.to_csv(index=False)
        st.download_button("Download CSV", csv, "retrieval_results.csv", "text/csv")

        st.subheader("t-SNE projection")
        chart = compute_tsne_plot(emb_sub, q_emb, df_sub)
        st.altair_chart(chart, use_container_width=True)


if __name__ == "__main__":
    main()
