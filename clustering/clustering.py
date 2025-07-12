import pandas as pd
from hdbscan import HDBSCAN
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import umap.umap_ as umap
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, TextGeneration
from copy import deepcopy
from transformers import pipeline

def topic_differences(model, original_topics, nr_topics=5):
    df = pd.DataFrame(columns=["Topic", "Original", "Updated"])
    for topic in range(nr_topics):
        og_words = " | ".join(list(zip(*original_topics[topic]))[0][:5])
        new_words = " | ".join(list(zip(*model.get_topic(topic)))[0][:5])
        df.loc[len(df)] = [topic, og_words, new_words]
    return df

# Load the dataset
dataset = load_dataset("maartengr/arxiv_nlp")["train"]
abstracts = dataset["Abstracts"][:]
titles = dataset["Titles"][:]

# Embed the texts
embedding_model = SentenceTransformer("thenlper/gte-small")
embeddings = embedding_model.encode(abstracts, show_progress_bar=True)

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

# Topic modeling for the embeddings
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    verbose=True,
).fit(abstracts, embeddings)
print(topic_model.get_topic_info())

fig = topic_model.visualize_documents(
    titles,
    reduced_embeddings=embeddings,
    width=1200,
    hide_annotations=True,
)
fig.update_layout(font=dict(size=16))
topic_model.visualize_barchart()

# Add additional blocks for the topic model
original_topics = deepcopy(topic_model.topic_representations_)
representation_model = KeyBERTInspired()
topic_model.update_topics(
    abstracts,
    representation_model=representation_model,
)
res_optimized = topic_differences(topic_model, original_topics)

representation_model = MaximalMarginalRelevance()
topic_model.update_topics(abstracts, representation_model=representation_model)
res_filtered = topic_differences(topic_model, original_topics)

prompt = prompt = """I have a topic that contains the following documents:
[DOCUMENTS]

The topic is described by the following keywords: '[KEYWORDS]'.

Based on the documents and keywords, what is this topic about?"""

generator = pipeline("text2text-generation", model="google/flan-t5-small")
representation_model = TextGeneration(
    generator, prompt=prompt, doc_length=50, tokenizer="whitespace"
)
topic_model.update_topics(abstracts, representation_model=representation_model)
res_generated = topic_differences(topic_model, original_topics)
print(res_generated)