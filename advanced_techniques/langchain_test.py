from langchain import PromptTemplate
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
template = """<s><|user|> {input_prompt}<|end|> <|assistant|>"""
prompt = PromptTemplate(
    template=template,
    input_variables=["input_prompt"]
)

basic_chain = prompt | llm
res = basic_chain.invoke(
    {
        "input_prompt": "Hi! My name is Maarten. What is 1 + 1? pls give me an answer with pure numbers no characters.",
    }
)
print(res)

