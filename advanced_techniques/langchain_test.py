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
res = llm.invoke("Am I cool?")
print(res)