from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryMemory
from langchain_community.llms import LlamaCpp

model_path = "Phi-3-mini-4k-instruct-fp16.gguf"
llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=-1,
    max_tokens=500,
    n_ctx=2048,
    seed=43,
    verbose=False
)




# # Test PromptTemplate
# template = """<s><|user|> {input_prompt}<|end|> <|assistant|>"""
# prompt = PromptTemplate(
#     template=template,
#     input_variables=["input_prompt"]
# )
# basic_chain = prompt | llm
# res = basic_chain.invoke(
#     {
#         "input_prompt": "Hi! My name is Maarten. What is 1 + 1? pls give me an answer with pure numbers no characters.",
#     }
# )
# print(res)




# # Test multiple prompts
# template = """<s><|user|>Create a title for a story about {summary}. Only return the title.<end><|assistant|>"""
# title_prompt = PromptTemplate(
#     template=template,
#     input_variables=["summary"],
# )
# title = LLMChain(
#     llm=llm,
#     prompt=title_prompt,
#     output_key="title",
# )
# # res = title.invoke({
# #     "summary": "a girl that lost her mother"
# # })
# # print(res)
# template = """<s><|user|>Describe the main character of a story about {summary} with the title {title}.
# Use only two sentences.<end>
# <|assistant|>"""
# character_prompt = PromptTemplate(
#     template=template,
#     input_variables=["summary", "title"],
# )
# character = LLMChain(
#     llm=llm,
#     prompt=character_prompt,
#     output_key="character",
# )
# # res = (title | character).invoke({
# #     "summary": "a girl that lost her mother"
# # })
# # print(res)
# template = """<s><|user|>Create a story about {summary} with the title {title}. The main character is:
# {character}. Only return the story and it cannot be longer than one paragraph.
# <end>
# <|assistant|>"""
# story_prompt = PromptTemplate(
#     template=template,
#     input_variables=["summary", "title", "character"],
# )
# story = LLMChain(
#     llm=llm,
#     prompt=story_prompt,
#     output_key="story",
# )
# llm_chain = title | character | story
# res = llm_chain.invoke("a girl that lost her mother")
# print(res["story"])




# Test Memory modules
# # Buffer Memory
# template = """<s><|user|>Current conversation: {chat_history}
#
# {input_prompt}<end>
# <|assistant|>"""
# prompt = PromptTemplate(
#     template=template,
#     input_variables=["chat_history", "input_prompt"],
# )
# memory = ConversationBufferMemory(
#     memory_key="chat_history",
# )
# llm_chain = LLMChain(
#     prompt=prompt,
#     memory=memory,
#     llm=llm,
# )
# res1 = llm_chain.invoke("Hi! My name is Maarten. What is 1 + 1?")
# print(res1)
# res2 = llm_chain.invoke("What's my name?")
# print(res2)

# # Windowed Conversation Buffer
# template = """<s><|user|>Current conversation: {chat_history}
#
# {input_prompt}<end>
# <|assistant|>"""
# prompt = PromptTemplate(
#     template=template,
#     input_variables=["chat_history", "input_prompt"],
# )
# memory = ConversationBufferWindowMemory(
#     k=1,
#     # k=3,
#     memory_key="chat_history",
# )
# llm_chain = LLMChain(
#     prompt=prompt,
#     memory=memory,
#     llm=llm,
# )
# res1 = llm_chain.invoke("Hi! My name is Maarten. What is 1 + 1?")
# print(res1)
# res2 = llm_chain.invoke("What's 3 + 3?")
# print(res2)
# res3 = llm_chain.invoke("What's my name?")
# print(res3)

# Conversation Summary
summary_template = """<s><|user|>Summarize the conversations and update
Current summary:
{summary}

new lines of conversation:
{new_lines}

New summary:<end>
<|assistant|>"""
summary_prompt = PromptTemplate(
    template=summary_template,
    input_variables=["new_lines", "summary"],
)
memory = ConversationSummaryMemory(
    memory_key="chat_history",
    llm=llm,
    prompt=summary_prompt,
)
template = """<s><|user|>Current conversation:{chat_history}
{input_prompt}<|end|>
<|assistant|>"""

prompt = PromptTemplate(
template=template,
input_variables=["input_prompt", "chat_history"]
)

llm_chain = LLMChain(
    prompt=prompt,
    memory=memory,
    llm=llm,
)
res1 = llm_chain.invoke({"input_prompt": "Hi! My name is Maarten. What is 1 + 1?"})
print(res1)
res2 = llm_chain.invoke({"input_prompt": "What is my name?"})
print(res2)
res3 = llm_chain.invoke({"input_prompt": "What was the first question I asked?"})
print(res3)