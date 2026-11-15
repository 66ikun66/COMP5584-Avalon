# 改动记录

## 2026-11-15 — 投票 prompt 重写、思考模式接入、观战链接

### 1. 前端：顶栏常驻身份徽标

之前只有中间栏「你的密令」卡片显示角色，容易漏看（误以为自己是普通好人）。现在右上角
LLM 徽标左侧常驻一枚胶囊：好人绿色 `身份：梅林 · 好人`，坏人红色 `身份：刺客 · 坏人`。

- `static/index.html`：新增 `#role-badge`
- `static/app.js`：`render()` 里按 `private.role_cn / alignment` 渲染
- `static/style.css`：`.badge-role` 两套配色

### 2. 投票提问语重写（只改 prompt，不改规则）

原提示语只有一句「请对该队伍投票（注意：同一轮连续 5 次否决坏人直接获胜）」，只讲风险、
不要求判断，导致投票变成橡皮图章（实测首提直接通过率 92%）。

现在改为「先算再投」，并且**不写死规则**——好坏两版都要求权衡，没有固定答案：

- 好人版：队里有你怀疑的人就该反对；但反对要挑场合，无脑否每一支队伍会白白消耗机会；
  同时点明「只有连续 5 次否决才判负，单次否决几乎没有代价」，纠正对否决成本的误判。
- 坏人版：有队友就保护他进队；纯好人队且很可能成功时应该狙击；但票是公开记录，
  反复反对会暴露立场，有时为了伪装也得投赞成。

### 3. LLM 适配层：思考模式分层开启

`agent._BUDGET` 按决策类型给预算与是否思考：

| 决策 | max_tokens | 思考 |
|---|---|---|
| vote | 600 | ✗ |
| mission | 400 | ✗ |
| speech | 2500 | ✓ |
| team | 3000 | ✓ |
| assassination | 5000 | ✓ |

- `llm._extra_body()`：按 `base_url` 选参数名——DeepSeek 原生 `thinking.type`，
  OpenRouter `reasoning.enabled`。发错参数不会报错、只会被静默忽略，这是之前「关思考失效」的根因。
- `reasoning.max_tokens` 限制思考不超过预算的 60%。实测出现过 `reasoning_tokens=1508`
  而预算只有 1200 的情况，思考把输出空间吃光，JSON 直接截断、整局报废。
- `llm._call()`：捕获 `length limit`，降级为「关思考 + 加大预算」重跑；
  原来那段「截断后重试」的分支跑不到（异常先抛出），属于死代码。
- `llm._invoke()`：指数退避重试 3 次（原来 `max_retries=0`）。单次成功率 99.7% 时，
  一局 148 次串行调用连乘只剩约 36% 成功率，实测 3/8 局崩溃。
- 新增环境变量 `LLM_MAX_TOKENS`、`LLM_TIMEOUT`、`LLM_THINKING=off|on|smart`。
  默认 `off`，行为与此前的基线数据一致，不影响可比性。

### 4. `server.py`：支持固定 sid

新增 `?sid=xxx`：脚本与浏览器可以进入同一局，便于观战 / 自动演示。
顺带修了 `do_POST` 用 `self.path` 精确匹配导致带查询参数时 404 的问题。

### 5. 实验与分析脚本（独立目录，不入本仓库）

分析代码与对局数据放在与仓库同级的 `avalon_experiments/`，不与应用代码混在一起：

| 文件 | 作用 |
|---|---|
| `play.py` | 离线跑批，落盘每局完整轨迹（发言、投票、任务、决策） |
| `analyze.py` / `deep.py` / `detect.py` | 三层统计：投票橡皮图章、指控命中率、策略转移 |
| `ab.py` / `compare.py` | 多组 A/B 对照（基线 / 新 prompt / V4-Flash） |
| `measure_prompt.py` / `dump_prompt.py` | 量化 prompt 体积、导出真实 prompt 实例 |
| `probe_models.py` / `probe_thinking.py` | 模型延迟、思考 token、JSON 截断探测 |
| `live_play.py` | 在运行中的服务上自动替真人打完一局（演示用） |
| `REPORT.html` | 9 局行为分析报告 |
| `results*/` | 各组原始轨迹 |

## 2026-11-13 — 首次跑通

- 新建 conda 环境 `avalon`（Python 3.11），安装 `langgraph` 与 `langchain-openai`。
- `config.py` 重写：课程网关 `genai.comp.polyu.edu.hk` 需要校内网 / VPN，本机 443 不通；
  改为接 OpenRouter（`deepseek/deepseek-chat-v3.1`），Key 读环境变量或本地 key 文件。
- 端到端验证 7 人局：组队 → 发言 → 投票 → 任务 → 刺杀全程无报错，AI 生成 55 条自然中文发言。
