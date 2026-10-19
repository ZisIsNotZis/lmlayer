# lmlayer 状态

一句话：OpenAI 兼容的 LLM 增强层（透明反向代理），坐在任意讲 OpenAI API 的 LLM 服务前，补安全校验、服务端会话历史、Auto RAG/rerank、工具沙箱、token 配额与公平调度（`README.md:2`）。

## 当前状态

**未被声明为关闭，但自 2026-01-07 起无功能改动（as-of 2026-10-19）。**

- 状态行：`docs/project-status.md:10-21` —— 不再写 "Active. Core features implemented and runnable"，改为按提交历史表述：最后功能改动 2026-01-07，2026-09-30 的提交只动 README/Dockerfile/compose/冒烟测试/文档，`69497e9`（2026-10-15）为仓库元数据。判定依据是该文件内写明的 `git log -1 -- <path>` 与提交号 `805f072`、`57ddeab`、`498e298`、`6f5644c`、`69497e9`。
- 代码主体最后改动 2026-01-07（`app.py`、`core.py`、`tool.py`、`util.py`、`db/common.py` 各自最后提交均为 `805f072`/`57ddeab`）。
- 打包与文档最后改动 2026-09-30（`README.md`、`tests/smoke_import.py`、`docker-compose.yml` 为 `498e298`；`docs/project-status.md` 为 `6f5644c`）。
- 2026-10-15 只有一次仓库维护提交 `69497e9 chore: ignore local agent workspace and refresh repository metadata`，不涉及功能。
- 边界：本次只读仓内文件，未运行任何代码；"可运行"未在本次复跑，见 `FINDINGS.md` 的"未知"节。

## 最新进展落在哪个文件

- 状态与证据索引：`docs/project-status.md`（分类、状态、证据、边界四节）。
- 使用与部署说明：`README.md`；冒烟检查入口 `tests/smoke_import.py`。
- 能力实现：`core.py`（对话/嵌入/重排流水线、安全校验、成本记账）、`app.py`（路由与透传）、`tool.py`（工具执行）、`util.py`（优先锁）、`db/`（pgvector 与本地 numpy 两种后端）。

## 下一步

仓内自己写下的未完成项只有这两处，按原文照搬：

- 联网搜索工具仍是计划项：`README.md:27` —— "(Planned) Web search integration"。
- 检索与安全流水线尚无端到端测试，冒烟检查只断言 HTTP 表面：`docs/project-status.md:37-38`。工具执行隔离依赖部署方的专用 Docker 网络：`docs/project-status.md:38-39`。

除此以外，仓内没有记录任何排期、里程碑或后续计划（见 `FINDINGS.md` 的"未知"节）。
