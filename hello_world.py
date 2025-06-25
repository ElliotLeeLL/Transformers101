import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# model = AutoModelForCausalLM.from_pretrained(
#     "microsoft/Phi-3-mini-4k-instruct",
#     device_map="cuda",
#     torch_dtype="auto",
#     trust_remote_code=True
# )
# tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
#
# generator = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     return_full_text=False,
#     max_new_tokens=500,
#     do_sample=False,
# )
#
# message = [
#     {"role": "user", "content": "How many states there in the USA?"}
# ]
#
# output = generator(message)
# print(output[0]["generated_text"])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

prompt = "How many states there in the USA?"

token_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

output = model.generate(
    input_ids=token_ids,
    max_new_tokens=100,
)
print(tokenizer.batch_decode(output)[0])