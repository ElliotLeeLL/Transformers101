import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification, DataCollatorForTokenClassification, \
    TrainingArguments, Trainer, pipeline
import evaluate

def align_labels(examples):
    token_ids = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True
    )
    labels = examples["ner_tags"]
    updated_labels = []
    for index, label in enumerate(labels):
        # Map tokens to their respective word
        word_ids = token_ids.word_ids(batch_index=index)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # The start of a new word
            if word_idx != previous_word_idx:
                previous_word_idx = word_idx
                updated_label = -100 if word_idx is None else label[word_idx]
                label_ids.append(updated_label)
            # Special token is -100
            elif word_idx is None:
               label_ids.append(-100)
            # If the label is B-XXX we change it to I-XXX
            else:
                updated_label = label[word_idx]
                if updated_label % 2 == 1:
                    updated_label += 1
                label_ids.append(updated_label)
        updated_labels.append(label_ids)
    token_ids["labels"] = updated_labels
    return token_ids

def compute_metrics(eval_pred):
    seqeval = evaluate.load("seqeval")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=2)

    true_predictions = []
    true_labels = []

    for prediction, label in zip(predictions, labels):
        for token_prediction, token_label in zip(prediction, label):
            if token_label != -100:
                true_predictions.append([id2label[token_prediction]])
                true_labels.append([id2label[token_label]])
    results = seqeval.compute(
        predictions=true_predictions,
        references=true_labels,
    )
    return {"f1": results["overall_f1"]}

dataset = load_dataset("conll2003", trust_remote_code=True)

label2id = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-ORG": 3,
    "I-ORG": 4,
    "B-LOC": 5,
    "I-LOC": 6,
    "B-MISC": 7,
    "I-MISC": 8,
}
id2label = {index: label for label, index in label2id.items()}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model_id = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForTokenClassification.from_pretrained(
    model_id,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
).to(device)

example = dataset["train"][848]
tokenized = dataset.map(align_labels, batched=True)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

training_args = TrainingArguments(
    "model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    save_strategy="epoch",
    report_to="none"
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
trainer.train()
res = trainer.evaluate()
print(res)

trainer.save_model("ner_model")
token_classifier = pipeline(
    "token-classification",
    model="ner_model",
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1  # use GPU if available
)
prompt = "My name is Maarten."
model_res = token_classifier(prompt)
print(model_res)
