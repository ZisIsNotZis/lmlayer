# lmlayer - OpenAI-Compatible LLM Enhancement Layer
`lmlayer` is a lightweight yet comprehensive OpenAI-compatible enhancement layer for LLM (Large Language Model) servers (e.g., vllm/llama-server). It addresses critical functionality gaps in native LLM servers by adding enterprise-grade capabilities via a transparent proxy model—no modifications to underlying LLM services required.

## Core Features
### 🔒 Full-Lifecycle Safety Check
- Pre-processing: Regex-based & Model-based safety validation for user prompts
- Post-processing: Regex-based & Model-based safety validation for LLM responses
- Tool invocation: Regex-based & Model-based safety validation for tool commands

### 📝 Session & History Management
- Session-based automatic history retrieval (frontend-agnostic, reliable server-side control)
- Granular history metrics tracking:
  - Cache token count
  - PP (Prompt Processing) token count & timestamp
  - TG (Token Generation) token count & timestamp
- Customizable system prompt support

### 🧠 RAG & Embedding/Rerank
- Auto RAG (Automatic Retrieval-Augmented Generation)
- Auto Rerank for retrieved documents (improves relevance)
- Full embedding support (compatible with llama-server embedding endpoints)
- Native rerank support (compatible with llama-server rerank endpoints)

### 🔧 Secure Tool Invocation
- Python execution: Isolated via subprocess or Docker
- Bash command execution: Isolated via subprocess or Docker
- (Planned) Web search integration

### 📡 Streaming Output
- OpenAI-compatible streaming response
- Real-time safety check for streaming content

### ⚖️ Quota Management
- Token quota calculation & enforcement (prevent abuse)

### 🎛️ Fair Scheduling
- Resource semaphore (prevents resource exhaustion)
- Fair scheduling for concurrent requests (avoids starvation)
- Sticky session (boosts cache hit rate for repeated requests)

### 🌐 Transparent Proxy
- Passes through unrecognized HTTP headers/request arguments
- 100% OpenAI API compatibility (works with existing OpenAI clients)

### 🗄️ Flexible Database Support
- PostgreSQL (with pgvector for production-grade vector storage/retrieval)
- Local storage: Numpy-based cosine similarity (lightweight, no DB dependency for testing)

## Quick Start
### Prerequisites
- `llama-server` installed (for LLM/embedding/rerank inference)
- Docker (for PostgreSQL/pgvector deployment)
- Python 3.8+ with `uvicorn`, `asyncpg`, `pgvector`, `numpy` (core dependencies)
- Host network access (for Docker-PostgreSQL communication)

### Step 1: Start Underlying LLM Services
Launch llama-server instances for core LLM, embedding, and rerank (adjust model paths/parameters as needed):
```bash
# Main LLM server (default port: 8080)
llama-server -m Qwen3-0.6B-UD-Q4_K_XL.gguf --temp .7 --min-p 0 --top-p .8 --top-k 20 &

# Embedding server (port: 8081)
llama-server -m Qwen3-Embedding-0.6B-IQ4_XS.gguf --embedding --pooling cls --port 8081 &

# Rerank server (port: 8082)
llama-server -m Qwen3-Reranker-0.6B-IQ4_XS.gguf --rerank --port 8082 &
```

### Step 2: Start PostgreSQL with pgvector
Deploy a pgvector-enabled PostgreSQL instance for vector storage:
```bash
docker run --network host -d \
  -e POSTGRES_USER=root \
  -e POSTGRES_PASSWORD=root \
  -e POSTGRES_DB=postgres \
  pgvector/pgvector:pg18
```

### Step 3: Launch lmlayer
Set the database DSN and start the FastAPI-based lmlayer service:
```bash
DB=postgresql+asyncpg://root:root@127.0.0.1:5432/postgres uvicorn app:app --host 0.0.0.0 --port 8000
```

### Verify Deployment
Once all services are running, `lmlayer` will be accessible at `http://localhost:8000` and fully compatible with OpenAI API endpoints (e.g., `/v1/chat/completions`). You can test it with any OpenAI client (e.g., `openai` Python SDK) by pointing the base URL to `http://localhost:8000/v1`.

## Detailed Feature Explanation
### 1. Safety Check
lmlayer enforces end-to-end safety validation to block unsafe content:
- **Pre-check**: Validates user input before sending to the LLM (blocks harmful prompts via regex patterns or a dedicated safety model)
- **Post-check**: Scans LLM outputs before returning to the client (prevents unsafe responses)
- **Tool-check**: Validates tool commands (e.g., Python/Bash) to prevent malicious execution

### 2. Session History Management
Unlike frontend-managed history (unreliable), lmlayer:
- Maintains server-side session state tied to user/session IDs
- Automatically fetches relevant conversation history for each request
- Tracks token usage and timestamps for quota enforcement and analytics

### 3. Auto RAG & Rerank
- Automatically retrieves context from the vector database (pgvector/local) based on user queries
- Uses the rerank model to reorder retrieved documents (prioritizes high-relevance content)
- Injects context into the LLM prompt transparently (no client-side changes needed)

### 4. Fair Scheduling
Prevents resource starvation and ensures equitable access to the LLM:
- **Resource semaphore**: Limits concurrent requests to match the LLM server's capacity
- **Fair scheduling**: Prioritizes pending requests to avoid long tail delays
- **Sticky session**: Routes repeated requests from the same user to the same worker (improves cache hit rate)

### 5. Transparent Proxy Design
lmlayer acts as a "drop-in" proxy:
- Passes through all unrecognized headers/parameters to the underlying LLM server
- Maintains full compatibility with OpenAI API specs (works with existing clients like `openai` SDK)

## Production Deployment
### Install Python Dependencies
Create a `requirements.txt` file:
```txt
uvicorn[standard]
fastapi
asyncpg
pgvector
numpy
python-multipart
httpx
gunicorn  # For production process management
```
Install dependencies:
```bash
pip install -r requirements.txt
```

### Production-Grade Launch
Use `gunicorn` (instead of raw `uvicorn`) for process management:
```bash
DB=postgresql+asyncpg://root:root@127.0.0.1:5432/postgres \
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Key Production Considerations
- Use persistent volumes for PostgreSQL (avoid ephemeral Docker storage)
- Isolate Docker-based tool execution with dedicated networks
- Enable HTTPS via Nginx reverse proxy (for public access)
- Monitor resource usage (CPU/RAM) for the LLM/embedding/rerank servers

## Summary
- `lmlayer` is an OpenAI-compatible enhancement layer for native LLM servers (vllm/llama-server)
- Core value: Full-lifecycle safety, server-side session history, Auto RAG/rerank, and fair scheduling
- Easy to deploy: Works with existing llama-server instances, uses pgvector for vector storage, and maintains transparent OpenAI API compatibility