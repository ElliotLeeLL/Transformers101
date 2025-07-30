from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    AutoModelForCausalLM, DataCollatorForWholeWordMask, pipeline
import numpy as np


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tomatoes = load_dataset('rotten_tomatoes')
train_data, test_data = tomatoes['train'], tomatoes['test']

model_id = 'bert-base-cased'
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenized_train_data = train_data.map(preprocess_function, batched=True)
tokenized_test_data = test_data.map(preprocess_function, batched=True)
tokenized_train = tokenized_test_data.remove_columns(['label'])
tokenized_test = tokenized_test_data.remove_columns(['label'])

data_collator = DataCollatorForWholeWordMask(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)
training_args = TrainingArguments(
    "model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    save_strategy="epoch",
    report_to="none"
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator
)
tokenizer.save_pretrained("mlm")
trainer.train()
model.save_pretrained("mlm")

# mask_filler = pipeline(
#     "fill-mask",
#     model="bert-base-cased"
# )
# preds = mask_filler("What a horrible [MASK]!")
# for pred in preds:
#     print(f">>> {pred['sequence']}")

mask_filler = pipeline(
    "fill-mask",
    model="mlm"
)
preds = mask_filler("What a horrible [MASK]!")
for pred in preds:
    print(f">>> {pred['sequence']}")