## 项目启动与运行指南

### 1. 创建 conda 环境 `avalon`（依赖只装在此环境）

```bash
cd Avalon
conda env create -f environment.yml
```

> 说明：Agent 层基于 **LangGraph** 框架、LLM 经 **LangChain `ChatOpenAI`** 调用 OpenAI 兼容接口。


### 2. 启动

```bash
conda activate avalon 
python Avalon.py      
```
右上角徽标会显示当前 AI 后端（🤖 LLM ）。

### 3. 配置 LLM 智能体（必需）

配置大模型，后端为 **OpenAI 兼容**接口。

```python
# config.py
LLM_API_KEY  = ""
LLM_BASE_URL = ""
LLM_MODEL    = "" 
```

> **本机当前配置**：`config.py` 已指向 **OpenRouter**（`https://openrouter.ai/api/v1`，模型
> `deepseek/deepseek-chat-v3.1`），API Key 自动从 `lab/api_key_openrouter.txt` 读取；
> 也可用环境变量 `LLM_API_KEY / LLM_BASE_URL / LLM_MODEL` 覆盖。
> 注意 PolyU 网关 `genai.comp.polyu.edu.hk` 仅校内网/VPN 可达，本机直连会超时。

LLM 是项目运行的必要条件。启动后徽标会显示「🤖 LLM: \<model\>」；如果没有有效 API Key，服务不会启动。说明：
- Agent 用 LLM 推理：读取**全部发言（含你的自由文本** 做社交推理，发言自然、会真正"读懂"你的话。
- LLM 调用有网络延迟,已用三招缓解：**同类决策并发**、规则提示**缓存**、
  **后台 worker 推进**——`GET /api/state` 永远秒返，前端轮询会看到 AI 发言**逐条浮现**，不会卡死。
- **思考型模型**：DeepSeek v4 等会先生成一段思维链（`reasoning_content`）再作答。思考 token **计入
  `max_tokens`**，所以开思考必须同时给足输出预算，否则 JSON 会被思考挤断。三档可选：

| `LLM_THINKING` | 行为 | 适用 |
|---|---|---|
| `off`（默认） | 全局关思考，每次 1500 预算 | 跑批量实验、追求稳定与速度 |
| `on` | 全局开思考（配合 `LLM_MAX_TOKENS` 抬高下限） | 单局想看 AI 深思，慢但更聪明 |
| `smart` | **按决策分层**：投票/出牌不思考（600/400），发言思考（1200），组队（3000）、刺杀（4000） | 质量与成本的折中，推荐 |

```bash
# 分层思考（推荐）
LLM_THINKING=smart LLM_TIMEOUT=180 D:/miniconda3/envs/avalon/python.exe Avalon.py

# 全局开思考（务必给足预算与超时）
LLM_THINKING=on LLM_MAX_TOKENS=4000 LLM_TIMEOUT=240 D:/miniconda3/envs/avalon/python.exe Avalon.py
```

实测（V4-Flash，真实局面）：关思考约 3s/次、输出 ~50 token；开思考 5.8s/次（中位 4.7s、最慢 14.6s）、
输出 553 token（其中思考 505），JSON 6/6 可解析。预算表见 `agent.py` 的 `_BUDGET`。
连接类错误已内置 3 次指数退避重试（`llm._invoke`），不会因单次抖动废掉整局。

