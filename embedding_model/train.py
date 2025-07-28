import random

from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.trainer import SentenceTransformerTrainer
from tqdm import tqdm

embedding_model = SentenceTransformer('bert-base-uncased')

# train_data = load_dataset(
#     "glue",
#     "mnli",
#     split="train",
# ).select(range(50))
# train_dataset = train_data.remove_columns("idx")
# val_sts = load_dataset(
#     "glue",
#     "stsb",
#     split="validation",
# )
# train_loss = losses.SoftmaxLoss(
#     model=embedding_model,
#     sentence_embedding_dimension=embedding_model.get_sentence_embedding_dimension(),
#     num_labels=3
# )
# evaluator = EmbeddingSimilarityEvaluator(
#     sentences1=val_sts["sentence1"],
#     sentences2=val_sts["sentence2"],
#     scores=[score/5 for score in val_sts["label"]],
#     main_similarity="cosine",
# )

# # Cosine Similarity
# train_data = load_dataset(
#     "glue",
#     "mnli",
#     split="train",
# ).select(range(500))
# train_dataset = train_data.remove_columns("idx")
# mapping = {2:0, 1:0, 0:1}
# train_dataset = Dataset.from_dict({
#     "sentence1": train_dataset["premise"],
#     "sentence2": train_dataset["hypothesis"],
#     "label": [float(mapping[label]) for label in train_dataset["label"]],
# })
# val_sts = load_dataset(
#     "glue",
#     "stsb",
#     split="validation",
# )
# train_loss = losses.CosineSimilarityLoss(
#     model=embedding_model
# )
# evaluator = EmbeddingSimilarityEvaluator(
#     sentences1=val_sts["sentence1"],
#     sentences2=val_sts["sentence2"],
#     scores=[score/5 for score in val_sts["label"]],
#     main_similarity="cosine",
# )

# Multiple Negatives Ranking Loss
mnli = load_dataset(
    "glue",
    "mnli",
    split="train",
).select(range(500))
mnli = mnli.remove_columns("idx")
mnli = mnli.filter(lambda x: True if x["label"] == 0 else False)
train_dataset = {
    "anchor": [],
    "positive": [],
    "negative": [],
}
soft_negatives = mnli["hypothesis"]
random.shuffle(soft_negatives)
for row, soft_negatives in tqdm(zip(mnli, soft_negatives)):
    train_dataset["anchor"].append(row["premise"])
    train_dataset["positive"].append(row["hypothesis"])
    train_dataset["negative"].append(soft_negatives)
train_dataset = Dataset.from_dict(train_dataset)
val_sts = load_dataset(
    "glue",
    "stsb",
    split="validation",
)
train_loss = losses.MultipleNegativesRankingLoss(
    model=embedding_model
)
evaluator = EmbeddingSimilarityEvaluator(
    sentences1=val_sts["sentence1"],
    sentences2=val_sts["sentence2"],
    scores=[score/5 for score in val_sts["label"]],
    main_similarity="cosine",
)

args = SentenceTransformerTrainingArguments(
    output_dir="./model",
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    fp16=True,
    eval_steps=100,
    logging_steps=100,
)

trainer = SentenceTransformerTrainer(
    model=embedding_model,
    args=args,
    train_dataset=train_dataset,
    loss=train_loss,
    evaluator=evaluator,
)
trainer.train()
res = evaluator(embedding_model)
print(res)