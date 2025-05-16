# 1. 输入是字符串
# 2. 使用 llm 将这个字符串进行意图识别转换为可能的搜索关键字去搜索
# 3. 读取 url 整合

import os
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import FakeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.chat_models.tongyi import ChatTongyi


def get_embedding_model():
    return FakeEmbeddings(size=1352)


def text_splitter():
    return RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)


def simulate_llm_intent(input_str):

    llm = ChatTongyi(
        model="qwen-turbo",
        dashscope_api_key=os.getenv("DASH_SCOPE_API_KEY"),
        top_p=0.95,
        temperature=0.7,
    )
    prompt = f"请根据以下用户输入内容，提取最适合互联网搜索的关键词和相关搜索词。请优先考虑用户搜索习惯、搜索意图和常用表达，目标是生成一套有助于精准定位信息的、简明相关的关键词。只返回关键词列表，用空格分隔，不要任何引导语或解释。用户输入：{input_str}"
    response = llm.invoke(prompt)
    return response.content.strip()


class WebSearchLoader:
    """
    互联网搜索与内容加载类
    输入字符串，自动意图识别、搜索、抓取并切分网页内容
    """

    def __init__(self, input_str):
        # 输入字符串
        self.input_str = input_str
        # 关键词
        self.keywords = simulate_llm_intent(input_str)
        # 搜索对象
        self.search = DuckDuckGoSearchResults(output_format="list")
        # 搜索结果
        self.results = self.search.invoke(self.keywords)
        # 提取URL
        self.urls = [item["link"] for item in self.results if "link" in item]
        # 网页加载器
        self.loader = UnstructuredURLLoader(urls=self.urls)
        # 加载网页内容
        self.data = self.loader.load()
        # 文本切分
        self.all_splits = text_splitter().split_documents(self.data)

    def get_splits(self):
        """
        获取切分后的文档内容
        """
        return self.all_splits

    def get_urls(self):
        """
        获取搜索到的URL列表
        """
        return self.urls


def search_urls_by_intent(input_str):
    loader = WebSearchLoader(input_str)
    all_splits = loader.get_splits()
    vectorstore = Chroma.from_documents(
        documents=all_splits, embedding=get_embedding_model()
    )
    docs = vectorstore.similarity_search(input_str)

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_template(
        """请按照以下步骤对提供的文档内容进行分析和总结：

**思考/分析步骤 (Reasoning):**
1.  **理解主旨:** 深入阅读文档，确定整体讨论的主题和背景信息。
2.  **识别核心观点:** 提取文档中最主要的论点、结论或观点。
3.  **筛选关键信息:** 从文档中找出支持核心观点的重要证据、数据、事实或关键细节。
4.  **组织结构:** 将提取的主题、观点和关键信息进行逻辑梳理，构思一个清晰、条理分明的总结结构。
5.  **构思表达:** 考虑如何用简明、准确的中文将上述信息表达出来，突出重点。

**输出动作 (Action):**
根据上述分析，生成一份使用 Markdown 格式排版的总结。
*   使用 Markdown 标题（# 或 ##）表示主要主题或分节。
*   使用列表（- 或 *）列出核心观点或关键点。
*   可以使用加粗（**文本**）突出重要信息。
*   总结内容必须条理清晰、语言简明，并突出关键信息。

**最终输出要求:**
**只输出 Markdown 格式的总结内容，严禁包含任何额外的文字、说明、引导语或对步骤的描述。**

文档内容如下：
{docs}"""
    )
    llm = ChatTongyi(
        model="qwen-turbo",
        dashscope_api_key=os.getenv("DASH_SCOPE_API_KEY"),
        top_p=0.46,
        temperature=0.2,
    )

    chain = {"docs": format_docs} | prompt | llm | StrOutputParser()

    docs = vectorstore.similarity_search(input_str)

    v = chain.invoke(docs)

    return v.content


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    input_str = "flask 是什么怎么学习?"
    result = search_urls_by_intent(input_str)
    print(result)
    with open("output.md", "w", encoding="utf-8") as f:  # 写入文件，编码为utf-8
        f.write(result)
