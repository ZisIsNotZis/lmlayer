# lmlayer 已确立的事实与结论

本文件只搬运本仓库内已有的事实，不新增实验、不新增数字、不做质量评价。每条给出出处（文件:行）。

状态定义：**成立** = 仓内代码/测试直接支撑；**部分** = 仓内只有文字声明，没有对应的可执行验证；**已撤回** = 仓内文件记录该结论已被回退或不再成立。

## 已确立的结论

| # | 结论 | 关键数字 | 出处 | 状态 |
|---|---|---|---|---|
| 1 | 服务对外只注册 3 个 OpenAI 端点：`/v1/chat/completions`、`/v1/embeddings`、`/v1/rerank`，其余路径走通配透传 | 3 个具名端点 + 1 条通配路由 | `app.py:46`、`app.py:60`、`app.py:70`、`app.py:80` | 成立 |
| 2 | 冒烟检查以这 3 条端点存在与否作为通过条件，失败即非零退出 | 3 条路由 | `tests/smoke_import.py:20`、`tests/smoke_import.py:23-25` | 成立 |
| 3 | 透传时只剔除 `content-length` 与 `content-encoding` 两个逐跳头，其余头原样转发 | 2 个头 | `app.py:23-24` | 成立 |
| 4 | 安全校验有两条通道：正则（`SAFE_RE`）与可选的外部安全模型端点（`SAFE_MODEL`，空则不启用） | 1 条正则 + 1 个可配端点 | `core.py:163-164`、`core.py:381-382` | 成立 |
| 5 | 正则通道同时用于入口改写、流式输出逐块检查与最终输出改写 | 3 处使用点 | `core.py:191`、`core.py:243`、`core.py:265` | 成立 |
| 6 | 检索默认取 5 条 embedding 候选，重排默认保留 2 条 | 5 / 2 | `core.py:167-168`、`core.py:194`、`core.py:199` | 成立 |
| 7 | 成本按用户累计并按指数衰减，时间因子默认等于 3600 秒半衰期 | 半衰期 3600 s | `core.py:117-126`、`core.py:161` | 成立 |
| 8 | 并发控制是按实例并行度初始化的优先锁（堆优先队列）；等待者以 `Future` 挂起，释放时按优先级唤醒 | 每个模型条目 1 个锁 | `core.py:131`、`util.py:17-44` | 成立 |
| 9 | 工具执行统一走子进程，硬超时 60 秒，超时后杀进程并返回 `'timeout'` 作为退出码 | 60 s | `tool.py:40`、`tool.py:43-47` | 成立 |
| 10 | 默认工具集中 `python`/`bash` 跑在 `docker run --rm ipython/ipython` 容器内，`docker/python`/`docker/bash` 跑在宿主 | 4 个工具名 | `tool.py:50-53` | 成立 |
| 11 | pgvector 向量列维度由环境变量决定，默认 1024，表名按维度后缀命名 | 1024 维 | `db/common.py:13`、`db/common.py:55`、`db/common.py:60` | 成立 |
| 12 | 无数据库时使用 numpy 本地后端，对外提供与 pg 后端同名同签名的接口 | 5 个同名接口 | `db/local.py:40-59` 对照 `db/pg.py:61-78` | 成立 |
| 13 | 嵌入输入按块切分后再批量请求，默认块长由 `CHUNK` 控制 | 默认 1024 | `util.py:46`、`core.py:166`、`core.py:326` | 成立 |
| 14 | 仓库是 AGPL-3.0 | 1 个许可证 | `docs/project-status.md:39`、`LICENSE` | 成立 |

## 负结果

| # | 结论 | 关键数字 | 出处 | 状态 |
|---|---|---|---|---|
| N1 | 检索与安全流水线没有端到端测试；唯一的自动化检查是 `tests/smoke_import.py`，只断言 HTTP 表面存在 | 1 个测试文件 | `docs/project-status.md:37-38` | 成立 |
| N2 | 工具执行隔离不由代码保证，取决于部署方是否使用专用 Docker 网络 | — | `docs/project-status.md:38-39` | 成立 |
| N3 | "100% OpenAI API 兼容"只有 README 声明，仓内没有任何一致性测试或对拍脚本支撑 | — | `README.md:43`（声明）；反证：`docs/project-status.md:37-38`（仅 1 个表面级测试） | 部分 |

## 已知问题（仓内证据不足以下结论）

| # | 问题 | 为什么是问题 | 缺什么证据 | 出处 |
|---|---|---|---|---|
| K1 | 工具名与执行位置方向相反：`python`/`bash` 在 `docker run --rm ipython/ipython` 里执行，`docker/python`/`docker/bash` 在服务自身进程所在的环境里执行（不加沙箱） | 前缀 `docker/` 指向容器，实际却是不隔离的那一侧。若部署方按字面理解为"更隔离"而设 `TOOL=docker/python`，模型的命令会直接落在服务进程所在的机器上 | 需 owner 说明或一份写明命名约定的文档。仓内两种读法都自洽：(a) 命名笔误，`docker/` 本意是 `host/`；(b) 有意——服务按 `docker-compose.yml` 部署在容器里，`docker/X` 表示"就在这个 docker 里跑"，`X` 表示"再套一层 `docker run`"。`README.md:25-26` 只说明两种隔离模式存在，没写工具名与模式的对应关系 | `tool.py:50-53`、`core.py:169`、`README.md:25-26`、`be0f4ee`（引入这两个工具名的提交） |

补充事实（可直接核对）：默认工具集是 `TOOL` 环境变量的默认值 `'python bash'`（`core.py:169`），即默认暴露给模型的是容器内执行的那两个；不隔离的 `docker/python`、`docker/bash` 只有部署方显式写进 `TOOL` 才会启用。`tool.py:50-53` 四个条目的 `description` 文本与工具名同样不对应（`bash`、`docker/bash` 的描述都写 "Run code in python"）。

## 撤回 / 回退记录

暂无：本仓库内没有任何文件记录过被撤回或回退的结论。已核对 `docs/project-status.md`、`README.md` 与 `git log` 的提交标题，均无回退类记录。

## 未知（仓内无法确定，不做推断）

- 冒烟检查当前是否通过：`tests/smoke_import.py:10-11` 约定了退出码含义，但本任务禁止运行项目，未复跑，因此未知。
- 真实下游模型（llama-server/vLLM/SGLang/TGI/Ollama）的兼容性：`README.md:2` 点名了这些后端，仓内无对应测试或记录。
- 检索质量、重排效果、安全正则的漏拦/误拦率：仓内没有任何指标或数据文件。
- 生产并发行为（优先锁在真实负载下的表现）：`util.py:17-44` 是唯一实现，仓内无压测记录。
