## 1. 项目设计说明

- **后端**：Python 标准库 `http.server`，集中管理游戏状态、角色分配、规则校验，并驱动各 Agent。
  状态机驱动整局流程；每次真人操作后自动推进所有 AI，直到再次需要真人或游戏结束。
- **Agent 层（核心）**：每名 AI = 一个 `AvalonAgent`，其**感知 → 推理 → 行动**闭环用 **LangGraph StateGraph**
  建模（节点：`perceive` 渲染上下文 → `reason` 调 LLM → `act` 校验/落库；无需推理的分支由 `perceive` 短路）。
  把"公开信息（含全部发言转录）+ 自己的合法私有信息"渲染为上下文，经 LangChain `ChatOpenAI`（OpenAI 兼容接口）以 **JSON Schema / 工具** 形式产出结构化决策，并维护跨回合的私有记忆笔记。详见 [agent.py](agent.y) / [llm.py](llm.py)。
- **前端**：原生 HTML + CSS + JavaScript。轮询 `GET /api/state` 渲染，
  通过 REST 接口提交真人操作。
- **信息隔离（贯穿到 LLM 层）**：所有玩家真实角色只存在于后端。每个玩家/每个 Agent 只持有自己的 `knowledge`
  （规则允许其知道的信息）。**Agent 的 system 提示只注入它本人的合法机密**，绝不含他人身份；
  Agent 的私有推理只留服务端、不下发。`GET /api/state` 只返回公开信息 + 真人本人的合法私有信息，
  **游戏结束前绝不下发其他人的身份**。

## 2. 游戏规则与角色配置（5~10 人局）

开局弹窗选择人数（5~10），不同人数采用如下标准好/坏人配置（定义见 [roles.py](roles.py) `ROLE_SETUPS`）：

| 人数 | 好人配置 | 坏人配置 |
|---|---|---|
| 5 | 梅林·派西维尔·忠臣×1 | 刺客·莫甘娜 |
| 6 | 梅林·派西维尔·忠臣×2 | 刺客·莫甘娜 |
| 7 | 梅林·派西维尔·忠臣×2 | 刺客·莫甘娜·奥伯伦 |
| 8 | 梅林·派西维尔·忠臣×3 | 刺客·莫甘娜·莫德雷德 |
| 9 | 梅林·派西维尔·忠臣×4 | 刺客·莫甘娜·莫德雷德 |
| 10 | 梅林·派西维尔·忠臣×4 | 刺客·莫甘娜·莫德雷德·奥伯伦 |

**角色与已知信息**

| 角色 | 阵营 | 已知信息 |
|---|---|---|
| 梅林 Merlin | 好 | 看到坏人（**莫德雷德除外**，不含具体身份） |
| 派西维尔 Percival | 好 | 看到梅林与莫甘娜两个"梅林候选"，无法区分 |
| 忠臣 LoyalServant | 好 | 无额外信息 |
| 刺客 Assassin | 坏 | 与莫甘娜/莫德雷德互知（不含奥伯伦）；终局负责刺杀 |
| 莫甘娜 Morgana | 坏 | 与刺客/莫德雷德互知；在派西维尔眼中伪装成梅林 |
| 莫德雷德 Mordred | 坏 | 与刺客/莫甘娜互知；**梅林看不见他**，可大胆伪装好人 |
| 奥伯伦 Oberon | 坏 | 不认识队友，队友也不认识他 |

**任务配置**（每轮出任务人数 / 判负所需失败票，定义见 [game.py](game.py) `MISSION_CONFIG`）

| 人数 | 第1轮 | 第2轮 | 第3轮 | 第4轮 | 第5轮 |
|---|---|---|---|---|---|
| 5 | 2 | 3 | 2 | 3 | 3 |
| 6 | 2 | 3 | 4 | 3 | 4 |
| 7 | 2 | 3 | 3 | 4·**需2票** | 4 |
| 8 | 3 | 4 | 4 | 5·**需2票** | 5 |
| 9 | 3 | 4 | 4 | 5·**需2票** | 5 |
| 10 | 3 | 4 | 4 | 5·**需2票** | 5 |

> 5/6 人局任意一轮 1 张失败票即判负；7~10 人局第 4 轮需 2 张失败票才判负，其余轮 1 张即可。

**流程**：随机分配角色/座位/首任队长 → 队长选人 → **发言/指控阶段（按座位序从队长起，每人发言一次，可附带指控）**
→ 全体投票（过半通过，平票=否决）→ 通过则入选成员出任务（好人只能成功，坏人可成功/失败）→ 任务结果（含失败票数）公开但不暴露投票人。
**胜负**：好人 3 个任务成功 → 进入刺杀；坏人破坏 3 个任务 → 坏人胜；同一轮连续 5 次组队被否 → 坏人胜；
刺客刺中梅林 → 坏人胜，否则好人胜。

## 3. AI 智能体决策逻辑（Agent 架构）

### 3.1 Agent 闭环（`agent.py` + `llm.py`）—— 项目核心

每名 AI 玩家是一个 `AvalonAgent`，通过 LLM 完成
（`decide_team / decide_vote / decide_mission / decide_speech / decide_assassination`）。内部是一张
**LangGraph 状态图**（`perceive → reason → act`，无需推理的分支短路到结束），承载"感知→记忆→推理→行动"：

1. **感知**：把公开信息（任务结果、组队/投票历史、**全部发言转录，含真人自由文本**）+ 自己的合法私有信息
   渲染为上下文。玩家对 LLM 一律用 1-based「N号」表示，减少歧义。
2. **记忆**：每个 Agent 维护一段**持久私有笔记 `notes`**，每次决策由模型自行更新，形成跨回合的连续心智。
3. **推理 + 行动**：经 LangChain `ChatOpenAI` 调用 LLM（OpenAI 兼容 / DeepSeek），用 **JSON 输出模式** 拿到稳定的结构化 JSON
   （含私有 `reasoning`、更新后的 `notes`、以及该决策的动作字段）。`reasoning` 仅留服务端、不下发。
4. **显式失败**：未配置 LLM 时服务启动即报错；LLM 调用失败或结构化输出非法时显式报错，不使用本地替代策略。

信息隔离贯穿到 LLM 层：**Agent X 的 system 提示只包含 X 本人的合法机密**（如梅林看到的坏人、派西维尔的候选），
绝不含他人真实身份。LLM 调用通过 prompt caching 复用规则提示；投票/任务等同类决策在 LLM 模式下并发执行以降低延迟。

LLM Agent 的五类结构化决策为：组队、投票、任务出票、发言/指控和刺杀。每次决策都会接收公开历史与该角色合法的私有信息，并通过 JSON 输出完成动作校验；模型调用失败或输出不符合当前动作 schema 时，服务端会保留明确错误，不伪造成功决策。

## 4. 前端页面设计（`static/`）

三栏布局：左（玩家列表 + 任务进度/比分/否决计数）、中（行动区 + 你的秘密信息）、右（历史投票记录 + 游戏日志）。
右栏顶部新增 **💬 发言/指控** 聊天面板，实时显示所有玩家发言与指控气泡。
行动区按当前阶段切换：队长选人（点选）/ **发言（自由文本框，可发送或跳过）** / 投票（赞成·反对）/
出任务（成功·失败，好人禁用失败）/ 刺杀（点选目标）/ 结算横幅。玩家列表对真人合法可见信息做高亮（梅林看到的坏人、派西维尔的梅林候选），
游戏结束才公开全部角色。

## 5. 后端接口设计（REST / JSON）

| 方法 | 路径 | 入参 | 说明 |
|---|---|---|---|
| GET | `/api/state` | — | 公开状态 + 真人合法私有信息（并自动推进 AI） |
| POST | `/api/new_game` | `{}` | 重开一局 |
| POST | `/api/team` | `{team:[pid,...]}` | 真人作为队长提交队伍 |
| POST | `/api/speak` | `{text:str}` | 真人在发言阶段发言（留空=跳过） |
| POST | `/api/vote` | `{approve:bool}` | 真人投票 |
| POST | `/api/mission` | `{action:"success"\|"fail"}` | 真人出任务（好人传 fail 会被拒） |
| POST | `/api/assassinate` | `{target:pid}` | 真人（刺客）刺杀 |

非法操作返回 `400 {"error": "..."}`。


## 6. 核心数据结构

- **player**：`{id, name, is_human, role}`（`role` 仅后端持有）。
- **knowledge[pid]**：`{role, alignment, known_evil:set, merlin_candidates:set, info_text:[...]}`
  —— 每个玩家"合法可知"的信息边界。
- **公开历史**：`proposals=[{round,attempt,leader,team,votes,approved}]`、
  `missions=[{round,team,fails,success}]`、`chat=[{round,attempt,pid,name,text,accuse}]`、
  `accuse_counts={pid:次数}`。
- **流程状态**：`phase∈{TEAM,DISCUSS,VOTE,MISSION,ASSASSIN,OVER}`、`round`、`leader`、`vote_track`、
  `proposed_team`、`discuss_order`/`discuss_idx`、`votes`、`mission_actions`、`winner`。

## 7. 启动与运行

### 7.1 创建 conda 环境 `avalon`（依赖只装在此环境）

```bash
cd Avalon
conda env create -f environment.yml
```

> 说明：Agent 层基于 **LangGraph** 框架、LLM 经 **LangChain `ChatOpenAI`** 调用 OpenAI 兼容接口。


### 7.2 启动

```bash
conda activate avalon 
python Avalon.py      
```
右上角徽标会显示当前 AI 后端（🤖 LLM ）。

### 7.3 配置 LLM 智能体（必需）

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

