from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from agents.graph import ADBAgentGraph
from retrieval.hybrid_retriever import HybridRetriever
from utils.config import settings
from utils.logger import setup_logger

# Setup logging
setup_logger()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Multi-agent RAG system for Android/ADB operations",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
retriever = HybridRetriever()
agent_graph = ADBAgentGraph()


class QueryRequest(BaseModel):
    query: str
    top_k: int | None = 5
    filters: dict | None = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    retrieved_docs: list[dict]
    query_type: str


@app.get("/")
def root():
    return {"message": "ADB Knowledge Assistant API", "version": "0.1.0", "status": "running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/query", response_model=QueryResponse)
async def query_knowledge(request: QueryRequest):
    """Main query endpoint"""

    try:
        logger.info(f"Received query: {request.query}")

        # Retrieve relevant documents
        retrieved_docs = retriever.retrieve(
            query=request.query, top_k=request.top_k, filters=request.filters
        )

        # Process through agent graph
        answer = agent_graph.query(user_query=request.query, retrieved_docs=retrieved_docs)

        return QueryResponse(
            query=request.query,
            answer=answer,
            retrieved_docs=retrieved_docs,
            query_type="determined_by_router",
        )

    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=settings.debug_mode)
