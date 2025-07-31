from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, TrainingArguments, pipeline
from datasets import load_dataset
from trl import SFTTrainer


def format_prompt(example):
    chat = example['message']
    prompt = template_tokenizer.apply_chat_template(chat, tokenize=False)
    return {"text": prompt}

template_tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')

dataset = (
    load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
        .shuffle(seed=42)
        .select(range(3000))
        .map(format_prompt)
)

model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=bnb_config)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = "<PAD>"
tokenizer.padding_side = "left"

peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "k_proj", "gate_proj", "v_proj", "up_proj", "q_proj", "o_proj", "down_proj"
    ]
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)


output_dir = "./results"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    num_train_epochs=1,
    logging_steps=10,
    fp16=True,
    gradient_checkpointing=True
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=training_arguments,
    max_seq_length=512,
    peft_config=peft_config,
)

trainer.train()

trainer.model.save_pretrained("TinyLlama-1.1B-qlora")

model = AutoPeftModelForCausalLM.from_pretrained(
    "TinyLlama-1.1B-qlora",
    low_cpu_mem_usage=True,
    device_map="auto",
)
merged_model = model.merge_and_unload()
prompt="""<|user|>
Tell me something about Large Language Models.</s>
<|assistant|>
"""

pipe = pipeline(
    "text-generation",
    model=merged_model,
    tokenizer=tokenizer,
)
print(pipe(prompt)[0]["generated_text"])