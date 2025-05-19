from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import FakeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
import re
from llm.llm_factory import get_llm
from llm.search import WebSearchLoader


llm = get_llm()


class SummarizeAgent:
    """总结Agent，支持多轮分层总结、风格定制、外部知识增强"""

    def __init__(
        self, llm, chunk_size=1000, chunk_overlap=100, style="简明", use_search=False
    ):
        # 初始化，llm为大模型实例，style为总结风格，use_search控制是否用搜索增强
        self.llm = llm
        self.style = style
        self.use_search = use_search
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        if use_search:
            self.search_tool = DuckDuckGoSearchResults()

    def _search_enhance(self, text):
        # 用DuckDuckGo搜索补充背景知识
        try:
            result = self.search_tool.run(text)
            return f"\n【相关背景】{result}\n"
        except Exception:
            return ""

    def _summarize_chunks(self, chunks, style):
        summaries = []
        for chunk in chunks:
            prompt = f"请用中文以{style}风格总结以下内容：\n{chunk}"
            if self.use_search:
                prompt += self._search_enhance(chunk[:100])
            resp = self.llm.invoke(prompt)

            # 获取内容并清理 <think> 标签
            if isinstance(resp, str):
                content = resp.strip()
            elif hasattr(resp, "content"):
                content = resp.content.strip()
            else:
                content = str(resp).strip()

            # 清除 <think> 标签及内容
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            content = re.sub(r'\s+', ' ', content)  # 可选：压缩多余空格

            summaries.append(content)
        
        return summaries

    def summarize(self, text=None, all_splits=None, style=None):
        # 主入口，支持自定义风格和两种输入模式
        style = style or self.style
        # 如果传入all_splits，直接用，否则分割text
        if all_splits is not None:
            # all_splits为Document对象列表或字符串列表
            if hasattr(all_splits[0], "page_content"):
                chunks = [doc.page_content for doc in all_splits]
            else:
                chunks = list(all_splits)
        elif text is not None:
            chunks = self.splitter.split_text(text)
        else:
            raise ValueError("必须提供text或all_splits参数")
        # 多轮分层总结，直到段数很少
        while len(chunks) > 3:
            summaries = self._summarize_chunks(chunks, style)
            # 每3段合并为一段，继续总结
            new_chunks = []
            for i in range(0, len(summaries), 3):
                merged = "\n".join(summaries[i : i + 3])
                new_chunks.append(merged)
            chunks = new_chunks
        # 最后一次总结，输出最终结果
        final_prompt = f"请将以下内容以{style}风格总结为一段中文总结：\n" + "\n".join(
            chunks
        )
        # 如果用搜索增强，补充背景
        if self.use_search:
            if text:
                final_prompt += self._search_enhance(text[:100])
            elif chunks:
                final_prompt += self._search_enhance(chunks[0][:100])
        final_resp = self.llm.invoke(final_prompt)
        if hasattr(final_resp, "content"):
            return final_resp.content
        return str(final_resp)


# 示例用法
agent = SummarizeAgent(llm, style="要点", use_search=True)
# result = agent.summarize("你的长文本内容")
# print(result)


loader = WebSearchLoader("flask 学习")
all_splits = loader.get_splits()
print("搜索到的URL：", loader.get_urls())
result = agent.summarize(all_splits=all_splits, style="要点")
print("最终总结：", result)