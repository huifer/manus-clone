from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os  

ollama_base_url = os.getenv("OLLAMA_BASE_URL", )


def get_llm(template=0.7,topp=0.3):
    llm = Ollama(base_url=ollama_base_url, model="qwen3:32b")  
    return llm