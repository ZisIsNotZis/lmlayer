```
llama-server -m Qwen3-0.6B-UD-Q4_K_XL.gguf --temp .7 --min-p 0 --top-p .8 --top-k 20 &
llama-server -m Qwen3-Embedding-0.6B-IQ4_XS.gguf --embedding --pooling cls --port 8081 &
llama-server -m Qwen3-Reranker-0.6B-IQ4_XS.gguf --rerank --port 8082 &
docker run --network host -dePOSTGRES_USER=root -ePOSTGRES_PASSWORD=root -ePOSTGRES_DB=postgres pgvector/pgvector:pg18
DB=postgresql+asyncpg://root:root@127.0.0.1:5432/postgres uvicorn app:app
```

* Protection
    * Pre
        * [x] Re
        * [x] Model
    * Post
        * [x] Re
        * [x] Model
    * Tool
        * [x] Re
        * [x] Model
* General
    * [x] System prompt
    * History
        * [x] User
        * [x] Cache
        * [x] Pp
        * [x] PpTime
        * [x] Think
        * [x] Assistant
        * [x] Tg
        * [x] TgTime
    * Now
        * [x] RAG
            * [x] Rerank
        * [x] User
        * [x] Think
        * [x] Response
        * Tool
            * [x] Python in Docker
            * [x] Bash in Docker
            * [ ] Search
* [x] Embedding
* [x] Rerank
* [x] Tool
* [x] Stream
* [x] Charging
* Scheduling
    * [x] Resource semaphore
    * [x] Fair scheduling
    * [x] Sticky session
* [x] Flexible input/output
* [x] pg
    * [x] pgvector