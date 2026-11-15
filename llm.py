"""
llm.py
======
LLM 客户端封装。为 Agent 提供"结构化决策"能力：给定 system / user 提示与一个工具(JSON Schema)，
让模型以该工具入参对应的 JSON 返回结构化结果。

后端：**OpenAI 兼容**（DeepSeek 等），通过 **LangChain 的 `ChatOpenAI`** 调用
`/chat/completions` + JSON 输出模式实现。配置来源（优先级：环境变量 > config.py）：
     LLM_API_KEY / LLM_BASE_URL / LLM_MODEL
   或 DeepSeek 习惯变量：DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL

说明：本文件是「模型客户端」；多智能体的**编排**（感知→推理→行动）见 agent.py，
那里用 **LangGraph** 把每个 Agent 的决策建模为一张状态图。

其它：
   AVALON_MODEL  覆盖模型名
   LLM_THINKING  on/off（默认 off）。DeepSeek v4 等是"思考型"模型，会先生成一大段思维链(reasoning_content)
                 再输出答案，既慢又容易把 token 预算耗尽导致 JSON 截断。默认关闭思考(更快更稳)；
                 设为 on 可开启思考（更"聪明"但更慢，需更大 max_tokens）。
"""

import json
import os
import threading
import time

_lock = threading.Lock()
_init_done = False
_cfg = None          # {"provider","api_key","base_url","model"} 或 None
_models = {}         # (max_tokens, thinking_off) -> ChatOpenAI（按需缓存，避免重复建客户端）
_thinking_unsupported = False     # 若某网关不认 thinking 参数，置位后不再发送


def _from_config(name, default=None):
    """读取 config.py 中的变量（文件不存在则返回 default）。"""
    try:
        import config
        return getattr(config, name, default)
    except Exception:
        return default


def _resolve():
    """探测可用后端配置；环境变量优先于 config.py。返回配置 dict 或 None。"""
    model_override = os.environ.get("AVALON_MODEL")

    # OpenAI 兼容（DeepSeek 等）
    api_key = (os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
               or _from_config("LLM_API_KEY"))
    if api_key:
        base = (os.environ.get("LLM_BASE_URL") or os.environ.get("DEEPSEEK_BASE_URL")
                or _from_config("LLM_BASE_URL") or "https://api.deepseek.com")
        model = (model_override or os.environ.get("LLM_MODEL")
                 or _from_config("LLM_MODEL") or "deepseek-chat")
        return {"provider": "openai", "api_key": api_key,
                "base_url": base.rstrip("/"), "model": model}

    return None


def _init():
    global _init_done, _cfg
    if _init_done:
        return
    with _lock:
        if _init_done:
            return
        _init_done = True
        _cfg = _resolve()


def is_available() -> bool:
    _init()
    return _cfg is not None


def provider_name() -> str:
    _init()
    if not _cfg:
        raise RuntimeError("未配置 LLM。请在 config.py 或环境变量中设置 LLM_API_KEY。")
    return _cfg["provider"]


def model_name() -> str:
    _init()
    if not _cfg:
        raise RuntimeError("未配置 LLM。请在 config.py 或环境变量中设置 LLM_API_KEY。")
    return _cfg["model"]


# ---------------------------------------------------------------------------
# 把工具的 JSON Schema 转成给模型的"请输出 JSON"指令（用于 JSON 输出模式）
# ---------------------------------------------------------------------------
def _schema_instruction(tool):
    props = tool.get("input_schema", {}).get("properties", {})
    required = set(tool.get("input_schema", {}).get("required", []))
    lines = []
    for key, spec in props.items():
        # JSON 模式下不索取 reasoning：它是自由长文本，极易把输出撑到 max_tokens 截断。
        # 模型仍会"隐式推理"，我们只取动作字段 + 一句短记忆。
        if key == "reasoning":
            continue
        t = spec.get("type", "string")
        if t == "array":
            it = spec.get("items", {}).get("type", "string")
            t = f"array<{it}>"
        flag = "必填" if key in required else "可选"
        desc = spec.get("description", "")
        enum = spec.get("enum")
        enum_note = f"；只能从 {enum} 中选择" if enum else ""
        lines.append(f'  "{key}" ({t}, {flag}): {desc}{enum_note}')
    return ("请只输出一个 JSON 对象（不要任何额外文字、不要代码块围栏、不要 reasoning 字段），字段如下：\n"
            + "\n".join(lines)
            + "\n注意：记忆字段(memo/belief_updates)务必从简；务必输出**完整且简短**的 JSON。")


def _minimal_instruction(tool):
    """只索取"必填动作字段"（去掉 reasoning 与记忆字段），用于截断后的极简重试。"""
    props = tool.get("input_schema", {}).get("properties", {})
    _meta = ("reasoning", "updated_notes", "belief_updates", "memo")
    req = [k for k in tool.get("input_schema", {}).get("required", [])
           if k not in _meta]
    if not req:                                   # 兜底：取第一个非 meta 字段
        req = [k for k in props if k not in _meta][:1]
    lines = []
    for k in req:
        spec = props.get(k, {})
        t = spec.get("type", "string")
        if t == "array":
            t = f"array<{spec.get('items', {}).get('type', 'string')}>"
        enum = spec.get("enum")
        enum_note = f"；只能从 {enum} 中选择" if enum else ""
        lines.append(f'  "{k}" ({t}): {spec.get("description", "")}{enum_note}')
    return ("严格只输出一个最小 JSON 对象，且**只含**以下字段，不要 reasoning、不要任何解释或多余字段：\n"
            + "\n".join(lines))


def _extract_json(text):
    """从模型文本中稳健提取首个 JSON 对象。"""
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s[:4].lower() == "json":
            s = s[4:]
    try:
        return json.loads(s)
    except Exception:
        pass
    # 退而求其次：截取第一个 { 到最后一个 }
    i, j = s.find("{"), s.rfind("}")
    if 0 <= i < j:
        try:
            return json.loads(s[i:j + 1])
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# OpenAI 兼容（DeepSeek）：LangChain ChatOpenAI + /chat/completions + JSON 输出模式
# ---------------------------------------------------------------------------
def _thinking_off():
    v = (os.environ.get("LLM_THINKING") or _from_config("LLM_THINKING") or "off")
    return str(v).lower() not in ("on", "1", "true", "yes")


def smart_mode():
    """LLM_THINKING=smart：**按决策分层**开思考（投票/出牌不思考，组队/发言/刺杀思考）。

    由 agent.py 的 _BUDGET 表决定每次调用用多少输出预算、要不要思考；
    未开启该模式时行为与原来完全一致（全局关思考 + 1500 预算）。
    """
    v = (os.environ.get("LLM_THINKING") or _from_config("LLM_THINKING") or "off")
    return str(v).lower() == "smart"



def _make_model(max_tokens, thinking_on=None):
    """构造（并按 max_tokens + 思考开关 缓存）一个 ChatOpenAI 客户端。

    thinking_on=None 表示沿用全局 LLM_THINKING 设置；显式传 True/False 可做
    「按决策分层」——例如组队/刺杀开思考、投票关思考（见 agent._BUDGET）。
    """
    from langchain_openai import ChatOpenAI
    if thinking_on is None:
        thinking_on = not _thinking_off()
    key = (max_tokens, thinking_on, _thinking_unsupported)
    m = _models.get(key)
    if m is not None:
        return m
    kwargs = dict(
        model=_cfg["model"],
        api_key=_cfg["api_key"],
        base_url=_cfg["base_url"],
        temperature=0.6,
        max_tokens=max_tokens,
        timeout=float(os.environ.get("LLM_TIMEOUT") or 90),
        max_retries=0,
        # JSON 输出模式：与原实现一致，强制返回单个 JSON 对象
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    # 思考开关必须**显式双向声明**：只"不发送"参数时，供应商的默认值不可控
    # （实测 OpenRouter 上 V4-Flash 有时默认关，导致 thinking=True 实际没思考）。
    # 参数名因网关而异：DeepSeek 原生端点 thinking.type=enabled/disabled，
    # OpenRouter 是 reasoning.enabled=true/false；发错了会被静默忽略或报 400。
    kwargs["extra_body"] = _extra_body(thinking_on, max_tokens)
    m = ChatOpenAI(**kwargs)
    _models[key] = m
    return m


def _extra_body(thinking_on, max_tokens=1500):
    """按 base_url 选择正确的思考开关参数名；网关不支持时返回 None。

    开思考时**同时限制思考预算**（OpenRouter 的 reasoning.max_tokens）：否则模型可能
    思考 1500+ token，把 max_tokens 吃光，content 变成空串直接整局报废（已实测踩到）。
    留 40% 预算给真正的 JSON 输出。
    """
    if _thinking_unsupported:
        return None
    if "openrouter" in str(_cfg.get("base_url") or "").lower():
        if not thinking_on:
            return {"reasoning": {"enabled": False}}
        cap = max(256, int(max_tokens * 0.6))
        return {"reasoning": {"enabled": True, "max_tokens": cap}}
    return {"thinking": {"type": "enabled" if thinking_on else "disabled"}}


# 可重试的瞬时故障（连接被重置、超时、上游 5xx）。这些重试不会改变游戏语义。
_TRANSIENT = ("connection", "timeout", "timedout", "apiconnection", "apitimeout",
              "internalserver", "serviceunavailable", "ratelimit")


def _invoke(model, msgs, tries=3):
    """对瞬时故障做指数退避重试；其它错误立即抛出。

    一局要 130+ 次调用，单次 0.3% 的失败率在连乘后会把整局打废（实测 3/8 崩溃），
    所以这里的重试是必要的工程保险，而非可选优化。
    """
    for i in range(tries):
        try:
            return model.invoke(msgs)
        except Exception as e:
            name = type(e).__name__.lower()
            msg = str(e).lower()
            transient = any(t in name for t in _TRANSIENT) or any(t in msg for t in ("connection", "timed out"))
            if not transient or i == tries - 1:
                raise
            time.sleep(1.5 * (2 ** i))



def _call(system, user_content, max_tokens, thinking=None):
    """单次调用；返回 (解析后的 dict 或 None, finish_reason)。"""
    global _thinking_unsupported
    from langchain_core.messages import HumanMessage, SystemMessage
    msgs = [SystemMessage(content=system), HumanMessage(content=user_content)]
    thinking_on = (not _thinking_off()) if thinking is None else bool(thinking)
    sent_param = not _thinking_unsupported          # 本次是否发送了思考开关参数
    try:
        resp = _invoke(_make_model(max_tokens, thinking_on), msgs)
    except Exception as e:
        # 思考把预算吃光时，langchain 会直接抛"length limit"而不是返回空内容。
        # 这里转成 (None,"length")，交给 decide() 降级重试（关思考再要一次动作）。
        if "length limit" in str(e).lower():
            return None, "length"
        # 某些网关不认 thinking / reasoning 参数 -> 记下并去掉该参数重试一次
        msg = str(e).lower()
        param_err = (getattr(e, "status_code", None) == 400
                     or e.__class__.__name__ == "BadRequestError"
                     or "thinking" in msg or "reasoning" in msg)
        if sent_param and param_err:
            _thinking_unsupported = True
            resp = _invoke(_make_model(max_tokens, thinking_on), msgs)
        else:
            raise
    content = resp.content if isinstance(resp.content, str) else str(resp.content)
    finish = (resp.response_metadata or {}).get("finish_reason")
    return _extract_json(content), finish


def decide(system: str, user: str, tool: dict, max_tokens: int = 1500, thinking=None):
    """用结构化方式拿决策；配置、调用或输出失败时抛出明确异常。

    max_tokens 默认放宽到 1500：JSON 模式下若被截断会导致整段 JSON 不可解析，
    宁可多给预算也要保证决策字段完整。
    thinking=True/False 可按决策覆盖思考开关；None 表示沿用全局 LLM_THINKING。
    开思考时务必把预算调高（思考 token 计入 completion），否则 JSON 会被挤断。
    """
    _init()
    if not _cfg:
        raise RuntimeError("未配置 LLM。请在 config.py 或环境变量中设置 LLM_API_KEY。")
    # 思考型模型在部分路由上会忽略"关闭思考"，把预算烧在 reasoning 上 -> 允许用环境变量抬高下限
    _env_mt = os.environ.get("LLM_MAX_TOKENS")
    if _env_mt:
        try:
            max_tokens = max(max_tokens, int(_env_mt))
        except ValueError:
            pass
    try:
        out, reason = _call(system, user + "\n\n" + _schema_instruction(tool), max_tokens, thinking)
        # 思考超预算被截断 -> 关掉思考重要一次。宁可这一次想得浅，也不能让整局崩掉。
        if out is None and reason == "length" and thinking is not False:
            out, reason = _call(system, user + "\n\n" + _schema_instruction(tool),
                                max(max_tokens, 1500), thinking=False)
        if out is not None:
            return out
        # 极少数情况下仍失败：思考型模型即便如此也可能截断 -> 加大预算再试一次
        out, reason = _call(system, user + "\n\n" + _minimal_instruction(tool),
                            max_tokens=max(max_tokens * 2, 2000), thinking=thinking)
        if out is None:
            raise RuntimeError(f"LLM 输出不可解析 (finish_reason={reason})")
        return out
    except Exception as e:
        if isinstance(e, RuntimeError) and str(e).startswith("LLM 输出不可解析"):
            raise
        raise RuntimeError(f"LLM 调用失败：{e}") from e
