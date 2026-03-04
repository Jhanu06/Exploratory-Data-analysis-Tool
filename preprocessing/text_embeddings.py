import pandas as pd
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

@st.cache_resource
def load_model(model_name='all-MiniLM-L6-v2'):
    """
    Load SentenceTransformer model.
    Cached to prevent reloading on every interaction.
    """
    return SentenceTransformer(model_name)

def generate_embeddings(df, text_column, batch_size=32):
    """
    Generate sentence embeddings for a text column in batches.
    Returns a Series of tuples (for hashability in pandas/streamlit).
    """
    model = load_model()
    
    # Ensure text column is string
    sentences = df[text_column].astype(str).tolist()
    
    # Generate embeddings
    # encode() handles batching internally if batch_size is provided
    embeddings = model.encode(sentences, batch_size=batch_size, show_progress_bar=True)
    
    # Convert numpy array to list of tuples
    # Embeddings are typically float32, we keep them as is
    embeddings_list = [tuple(emb) for emb in embeddings]
    
    return pd.Series(embeddings_list, index=df.index)

def reduce_dimensions(embeddings_series, n_components=2):
    """
    Reduce dimensionality of embeddings using PCA.
    embeddings_series: Series of tuples/lists containing vectors.
    Returns a DataFrame with PCA components.
    """
    # Convert series of tuples back to numpy array
    if embeddings_series.empty:
        return pd.DataFrame()
        
    X = np.array(embeddings_series.tolist())
    
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X)
    
    cols = [f'PCA_{i+1}' for i in range(n_components)]
    return pd.DataFrame(components, columns=cols, index=embeddings_series.index)
