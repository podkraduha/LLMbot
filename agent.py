import os
import re
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_community.chat_models import QianfanChatEndpoint
from langchain.tools import Tool
from langchain_experimental.tools import PythonREPLTool

# 1. Защита от Prompt Injection (Blacklist опасных команд)
DANGEROUS_PATTERNS = [
    r'\bos\b', r'\bsys\b', r'\bsubprocess\b', r'\bshutil\b',
    r'\b__import__\b', r'\bopen\s*\(', r'\beval\s*\(', r'\bexec\s*\('
]

def get_agent(df: pd.DataFrame, api_key: str, user_context: str = ""):
    # Для Qwen через DashScope
    import dashscope
    from langchain_community.llms import Tongyi

    llm = Tongyi(
        model="qwen-turbo",  # или "qwen-plus", "qwen-max"
        dashscope_api_key=api_key
    )

def sanitize_code(code: str) -> str:
    """Проверяет код, который хочет выполнить агент, на вредоносность."""
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, code):
            raise ValueError(f"Обнаружена потенциально опасная команда в коде: {pattern}")
    return code
