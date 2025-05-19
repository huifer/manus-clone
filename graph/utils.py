from duckduckgo_search import DDGS
import os
import httpx
import requests
from typing import Dict, Any, List, Union, Optional
from markdownify import markdownify
from langsmith import traceable
from langchain_community.utilities import SearxSearchWrapper
from langchain_core.tools import BaseTool


@traceable
def duckduckgo_search(
    query: str, max_results: int = 3, fetch_full_page: bool = False
) -> Dict[str, List[Dict[str, Any]]]:
    """
    使用DuckDuckGo进行网页搜索并返回格式化结果。

    使用DDGS库通过DuckDuckGo执行网页搜索。

    参数:
        query (str): 要执行的搜索查询
        max_results (int, 可选): 返回的最大结果数。默认值为3。
        fetch_full_page (bool, 可选): 是否从结果URL获取完整页面内容。
                                    默认为False。
    返回:
        Dict[str, List[Dict[str, Any]]]: 搜索响应，包含:
            - results (list): 搜索结果字典列表，每个包含:
                - title (str): 搜索结果标题
                - url (str): 搜索结果URL
                - content (str): 内容摘要/片段
                - raw_content (str 或 None): 若fetch_full_page为True则为完整页面内容，否则与content相同
    """
    try:
        with DDGS() as ddgs:
            results = []
            search_results = list(ddgs.text(query, max_results=max_results))

            for r in search_results:
                url = r.get("href")
                title = r.get("title")
                content = r.get("body")

                if not all([url, title, content]):
                    print(f"警告: DuckDuckGo返回结果不完整: {r}")
                    continue

                raw_content = content
                if fetch_full_page:
                    raw_content = fetch_raw_content(url)

                # 添加结果到列表
                result = {
                    "title": title,
                    "url": url,
                    "content": content,
                    "raw_content": raw_content,
                }
                results.append(result)

            return {"results": results}
    except Exception as e:
        print(f"错误: DuckDuckGo搜索异常: {str(e)}")
        print(f"详细错误类型: {type(e).__name__}")
        return {"results": []}


def fetch_raw_content(url: str) -> Optional[str]:
    """
    从URL获取HTML内容并转换为markdown格式。

    使用10秒超时，避免在慢速网站或大页面上卡住。

    参数:
        url (str): 要获取内容的URL

    返回:
        Optional[str]: 若成功则返回转换为markdown的内容，若获取或转换出错则返回None
    """
    try:
        # 创建带合理超时的客户端
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url)
            response.raise_for_status()
            return markdownify(response.text)
    except Exception as e:
        print(f"警告: 获取{url}完整页面内容失败: {str(e)}")
        return None


class DuckDuckGoSearchTool(BaseTool):
    """
    LangGraph工具接口封装，duckduckgo搜索工具
    """

    # 工具名称，Pydantic必填
    name: str = "duckduckgo_search"
    # 工具描述，Pydantic必填
    description: str = "使用DuckDuckGo进行网页搜索，支持返回摘要和完整页面内容"

    # 搜索结果数量
    max_results: int = 3
    # 是否抓取完整页面
    fetch_full_page: bool = False
    

    def _run(self, query: str) -> dict:
        # 兼容工具调用接口
        return duckduckgo_search(
            query, max_results=self.max_results, fetch_full_page=self.fetch_full_page
        )


if __name__ == "__main__":
    # 中文注释: 测试duckduckgo_search函数
    query = "OpenAI"
    print("测试duckduckgo_search函数：")
    result = duckduckgo_search(query, max_results=2, fetch_full_page=False)
    print(result)
