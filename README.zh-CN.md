<div align="right"><a href="README.md">English</a> | 简体中文</div>

<p align="center">
  <img src="assets/logo.svg" alt="lmlayer" width="560">
</p>

# lmlayer

**保留你的模型服务，补上它缺的那一层。**

`lmlayer` 是一个开箱即用的 OpenAI 兼容代理。它把安全校验、服务端会话记忆、自动 RAG、工具沙箱、Token 配额和公平调度放在任意 LLM 服务前面——llama.cpp、vLLM、SGLang、TGI、Ollama，或任何讲 OpenAI API 的服务都可以。上游服务一行都不用改。

## 为什么需要 lmlayer

大多数 OpenAI 兼容服务只负责“把模型跑起来”。真正围绕模型的运行层——持久会话、输入输出安全、检索增强、工具隔离、按用户计费、公平排队——通常要自己补。`lmlayer` 把这一层放在协议边界上，现有 OpenAI 客户端可以继续照常使用。

## 快速开始

需要 Python 3.11+ 和至少一个 OpenAI 兼容上游。PostgreSQL + pgvector 是可选的；未设置 `DB` 时，`lmlayer` 会使用内存中的 NumPy 后端。只有默认的 `python`/`bash` 工具需要 Docker。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

默认配置下，对话、嵌入和重排上游分别位于 `http://localhost:8080`、`:8081`、`:8082`。要换成别的地址，设置 `MODEL`：

```bash
MODEL='{"qwen":{"cost":0,"instances":{"http://localhost:8080":1}},"qwenEmb":{"cost":0,"instances":{"http://localhost:8081":1},"embedding":true},"qwenRank":{"cost":0,"instances":{"http://localhost:8082":1},"rerank":true}}' \
uvicorn app:app --host 0.0.0.0 --port 8000
```

服务启动后监听 `http://localhost:8000`，可以直接接收 OpenAI 风格的请求：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Hello"}]}'
```

### 使用持久化存储

启动带 pgvector 的 PostgreSQL，并把 DSN 传给 `lmlayer`：

```bash
docker run --network host -d --name lmlayer-db \
  -e POSTGRES_USER=root \
  -e POSTGRES_PASSWORD=root \
  -e POSTGRES_DB=postgres \
  pgvector/pgvector:pg18

DB=postgresql+asyncpg://root:root@127.0.0.1:5432/postgres \
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Docker Compose

```bash
cp .env.example .env
# 编辑 .env：DB 和 MODEL 里的地址必须能从容器内部访问。
docker compose up --build
```

`.env.example` 是按“在宿主机直接运行”写的（地址为 `127.0.0.1`）。用 Compose 时，把数据库主机改成 `db`，上游地址改成容器能访问到的地址。

## 现在能做什么

- **OpenAI 兼容端点**——`/v1/chat/completions`、`/v1/embeddings`、`/v1/rerank`，其余路径原样透传。
- **安全校验**——对用户输入、模型输出和每个流式分块做正则检查，可选接入安全模型。工具参数和工具输出目前不做校验。
- **服务端会话**——按用户和会话保存历史，支持会话分叉，并记录缓存、Prompt、生成等指标。
- **自动 RAG**——先嵌入查询，从 pgvector 或本地 NumPy 存储中召回候选，再重排并注入 Prompt。
- **工具循环**——Python 和 Bash 工具以子进程方式执行，硬超时 60 秒。默认的 `python`/`bash` 跑在 `docker run --rm ipython/ipython` 容器里；命名恰好相反的 `docker/python`/`docker/bash` 则直接跑在服务自身所在的环境里，没有任何隔离。
- **配额与成本记账**——按用户累计成本，指数衰减，并据此限制用量。
- **公平调度**——每个模型一把优先锁（优先级即该用户衰减后的成本），并按实例空闲程度选择上游。
- **流式输出**——SSE 流式返回，边生成边检查。
- **存储可切换**——生产用 PostgreSQL/pgvector，开发用内存后端。

## 架构

```
OpenAI 客户端
     │  /v1/chat/completions · /v1/embeddings · /v1/rerank · /*
     ▼
┌──────────────────────────── lmlayer ────────────────────────────┐
│  安全校验  →  RAG（嵌入 → 召回 → 重排）  →  配额 + 优先锁        │
│     →  请求上游  →  流式返回 / 工具循环  →  响应                 │
└───────────────┬───────────────────────────────┬─────────────────┘
                │                               │
      PostgreSQL + pgvector              OpenAI 兼容上游
      （或内存 NumPy）                   （llama.cpp、vLLM、SGLang……）
```

| 路径 | 职责 |
| --- | --- |
| `app.py` | FastAPI 路由、流式响应、透明透传 |
| `core.py` | 对话/嵌入/重排流水线、安全校验、RAG、成本与调度 |
| `tool.py` | 工具定义与子进程执行 |
| `util.py` | 优先锁和文本切块 |
| `db/` | PostgreSQL 与内存后端，共用一个接口 |
| `tests/smoke_import.py` | 导入应用并断言 OpenAI 兼容表面存在 |

## 配置

所有配置都走环境变量，完整带注释的列表见 [`.env.example`](.env.example)。最常改的几个：

| 变量 | 作用 |
| --- | --- |
| `DB` | 异步 PostgreSQL DSN；不设置则使用内存后端 |
| `MODEL` | 模型名到上游实例、成本、角色的 JSON 映射 |
| `EMBEDDING_DIM` | pgvector 维度，默认 `1024` |
| `EMBEDDING`、`RERANK` | 召回候选数与重排后保留数 |
| `SAFE_RE`、`SAFE_MODEL` | 安全正则与可选的安全模型端点 |
| `TOOL` | 暴露给模型的工具，默认 `python bash` |
| `PROMPT`、`PROMPT_RAG` | 系统提示词与 RAG 提示词模板 |

## 开发

运行冒烟检查。它会导入应用并确认三个 OpenAI 兼容路由存在，不会启动数据库，也不会连接模型服务：

```bash
python tests/smoke_import.py
```

项目是纯 Python，没有构建步骤。`docker-compose.yml` 提供本地 pgvector + 服务的组合，[`docs/project-status.md`](docs/project-status.md) 记录了当前实现状态和已知边界。

## 参与贡献

欢迎提 Issue 和 Pull Request。改动尽量聚焦，提交 PR 前跑一遍冒烟检查；行为或配置发生变化时，请同步更新文档。简短说明见 [`CONTRIBUTING.md`](CONTRIBUTING.md)。

## 路线图

**已实现**

- [x] OpenAI 兼容的对话、嵌入、重排端点与透传
- [x] 用户输入、模型输出和流式内容的安全校验
- [x] 服务端会话历史与会话分叉
- [x] 基于 pgvector 或内存 NumPy 的自动 RAG
- [x] Docker/宿主工具执行与硬超时
- [x] 按用户计费、配额限制与成本衰减
- [x] 基于优先级的公平调度与最空闲上游选择
- [x] 带增量安全检查的 SSE 流式输出

**计划中**

- [ ] 联网搜索工具
- [ ] 检索与安全流水线的端到端测试
- [ ] 粘性会话路由——`_STICK` 只有读取，没有任何地方写入
- [ ] 对工具参数与工具输出做安全校验
- [ ] 理清 `python`/`bash` 与 `docker/python`/`docker/bash` 的命名，让隔离模式一眼可辨
- [ ] 为工具隔离补一份加固部署说明

长期目标是让 `lmlayer` 成为应用与任意模型服务之间那层“看不见的运行层”：一个端点就已经具备安全、记忆、检索、工具、预算和公平访问，而模型服务本身始终可以替换。

## 许可证

[AGPL-3.0](LICENSE)。
