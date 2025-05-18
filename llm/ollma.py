from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os  

ollama_base_url = os.getenv("OLLAMA_BASE_URL", )

llm = Ollama(base_url=ollama_base_url, model="qwen3:32b")  # 

# 定义提示模板
prompt = PromptTemplate(
    input_variables=["question"], template="请用中文简要回答：{question}"
)

# 构建LangChain链
chain = LLMChain(llm=llm, prompt=prompt)

if __name__ == "__main__":
    # 示例问题
    question = "什么是人工智能？"
    # 执行链并输出结果
    result = chain.run({"question": question})
    print("Ollama回答：", result)
