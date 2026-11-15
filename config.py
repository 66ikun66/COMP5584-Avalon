"""本地 LLM 配置（OpenAI 兼容接口）。

默认走 OpenRouter，模型 deepseek/deepseek-chat-v3.1（也支持任何 OpenAI 兼容端点）。
API Key 优先级：环境变量 LLM_API_KEY > 本目录下 api_key.txt > 课程工作区的 key 文件。

⚠️ 不要把 key 写进代码或提交到仓库：api_key.txt 已在 .gitignore 中。
"""

import os
import pathlib

_HERE = pathlib.Path(__file__).resolve().parent
_KEY_FILES = [
    _HERE / "api_key.txt",                                   # 推荐：仓库内单独放一个文件（已忽略）
    pathlib.Path(r"E:\polyu_sem2\LLMAgent\lab\api_key_openrouter.txt"),  # 本机既有位置（可删）
]


def _read_key_file():
    for path in _KEY_FILES:
        try:
            key = path.read_text(encoding="utf-8").strip()
            if key:
                return key
        except Exception:
            continue
    return ""


LLM_API_KEY = os.environ.get("LLM_API_KEY") or _read_key_file()
LLM_BASE_URL = os.environ.get("LLM_BASE_URL") or "https://openrouter.ai/api/v1"
LLM_MODEL = os.environ.get("LLM_MODEL") or "deepseek/deepseek-chat-v3.1"
LLM_THINKING = "off"
