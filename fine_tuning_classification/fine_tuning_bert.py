from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
import numpy as np
from evaluate import load


def preprocess_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    load_f1 = load("f1")
    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
    return {"f1": f1}

tomatoes = load_dataset('rotten_tomatoes')
train_data, test_data = tomatoes['train'], tomatoes['test']

model_id = 'bert-base-cased'
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenized_train = train_data.map(preprocess_function, fn_kwargs={"tokenizer": tokenizer}, batched=True)
tokenized_test = test_data.map(preprocess_function, fn_kwargs={"tokenizer": tokenizer}, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

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

# Froze part of the model layers
for index, (name, params) in enumerate(model.named_parameters()):
    if index < 165:
        params.requires_grad = False

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)
trainer.train()
res = trainer.evaluate()
print(res)


