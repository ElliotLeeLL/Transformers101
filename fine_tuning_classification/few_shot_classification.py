from datasets import load_dataset
from setfit import sample_dataset, SetFitModel, SetFitTrainer

# Load dataset
tomatoes = load_dataset('rotten_tomatoes')
train_data, test_data = tomatoes['train'], tomatoes['test']

# Few-shot sampling
sampled_train_data = sample_dataset(
    train_data,
    num_samples=16,
    seed=42
)

# Load model
model = SetFitModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")

trainer = SetFitTrainer(
    model=model,
    train_dataset=sampled_train_data,
    eval_dataset=test_data,
    metric="f1",
    num_epochs=3,
    num_iterations=32,
    batch_size=16,
    column_mapping={"text": "text", "label": "label"},
)

trainer.train()
metrics = trainer.evaluate()
print(metrics)