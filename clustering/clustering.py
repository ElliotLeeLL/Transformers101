from hdbscan import HDBSCAN
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import umap.umap_ as umap

# Load the dataset
dataset = load_dataset("maartengr/arxiv_nlp")["train"]
abstracts = dataset["Abstracts"]
titles = dataset["Titles"]

# Embed the texts
embedding_model = SentenceTransformer("thenlper/gte-small")
embeddings = embedding_model.encode(abstracts[:], show_progress_bar=True)

# Reduce the dimensions of embeddings
umap_model = umap.UMAP(
    n_components=5,
    min_dist=0.0,
    random_state=43,
    metric="cosine"
)
embeddings = umap_model.fit_transform(embeddings)
print(embeddings.shape)

# Cluster the embeddings
hdbscan_model = HDBSCAN(
    min_cluster_size=50,
    metric="euclidean",
    cluster_selection_method="eom",
).fit(embeddings)

clusters = hdbscan_model.labels_
print(len(set(clusters)))