import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
)

pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=500,
    do_sample=True,
)

# message = [
#     {
#         "role": "user",
#         "content": "Create a funny joke about chickens"
#     }
# ]
#
# output = pipeline(message, temperature=2.0, top_p=0.5)

# #Add Potential Complexity of a Prompt
# persona = "You are an expert in Large Language models. You excel at breaking" \
#     "down complex papers into digestible summaries.\n"
#
# instruction = "Summarize the key findings of the paper provided.\n"
#
# context = "Your summary should extract the most crucial points that can help" \
#     "researchers quickly understand the most vital information of the paper.\n"
#
# data_format = "Create a bullet-point summary that outlines the method. Follow" \
#     "this up with a concise paragraph that encapsulates the main results.\n"
#
# audience = "The summary is designed for busy researchers that quickly need to" \
#     "grasp the newest trends in Large Language Models.\n"
#
# tone = "The tone should be professional and clear.\n"
#
# text = "MY TEXT TO SUMMARIZE"
#
# data = f"Text to summarize: {text}"
#
# query = persona + instruction + context + data_format + audience + tone + data

# print(query)

# Chain of thought
# cot_prompt = [
#     {
#         "role": "user",
#         "content": "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?"
#     },
#     {
#         "role": "assistant",
#         "content": "Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11."
#     },
#     {
#         "role": "user",
#         "content": "The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?"
#     }
# ]

# cot_prompt = [
#     {
#         "role": "user",
#         "content": "The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have? Let's think step-by-step."
#     }
# ]
# outputs = pipeline(
#     cot_prompt
# )
# print(outputs)

# # Tree of thought
# zeroshot_tot_prompt = [
#     {
#         "role": "user",
#         "content": "Imagine three different experts are answering "
#                    "this question. All experts will write down 1 step of their thinking, then share "
#                    "it with the group. Then all experts will go on to the next step, etc. If any "
#                    "expert realizes they're wrong at any point then they leave. The question is "
#                    "'The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, "
#                    "how many apples do they have?' Make sure to discuss the results."
#     }
# ]
zeroshot_tot_prompt = [
    {
        "role": "user",
        "content": "Imagine three different experts are answering "
                   "this question. All experts will write down 1 step of their thinking, then share "
                   "it with the group. Then all experts will go on to the next step, etc. If any "
                   "expert realizes they're wrong at any point then they leave. The question is "
                   "'The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, "
                   "how many apples do they have?' Make sure to discuss the results."
    }
]
output = pipeline(zeroshot_tot_prompt)
print(output[0]["generated_text"])