from fastapi import APIRouter, Depends, HTTPException

from src.services.query_service import GraphQuerying
from src.config.schemas import QueryRequest, SearchRequest, QueryResponse
from src.api.dependencies import get_querying_service

router = APIRouter()


@router.post("/full_hybrid", response_model=QueryResponse)
async def query_knowledge_graph(
    request: QueryRequest,
    querying_service: GraphQuerying = Depends(get_querying_service)
):
    """Full Hybrid Retrieval: Graph RAG + Qdrant Hybrid Search with LLM generation."""
    try:
        result = querying_service.response_detailed(
            query=request.query,
            top_k=request.top_k,
            threshold=request.threshold,
            graph_limit=request.graph_limit
        )
        
        return QueryResponse(
            answer=result["answer"],
            graph_context=result["graph_context"],
            chunk_context=result["chunk_context"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dense", response_model=QueryResponse)
async def dense_search(
    request: SearchRequest,
    querying_service: GraphQuerying = Depends(get_querying_service)
):
    """Dense Search: Dense vector search + LLM generation."""
    try:
        result = querying_service.dense_response(
            query=request.query,
            top_k=request.top_k,
            threshold=request.threshold
        )
        
        return QueryResponse(
            answer=result["answer"],
            graph_context=result["graph_context"],
            chunk_context=result["chunk_context"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/semantic_hybrid", response_model=QueryResponse)
async def semantic_hybrid_search(
    request: SearchRequest,
    querying_service: GraphQuerying = Depends(get_querying_service)
):
    """Hybrid Search: Qdrant hybrid (Dense + Sparse) + LLM generation."""
    try:
        result = querying_service.semantic_hybrid_response(
            query=request.query,
            top_k=request.top_k,
            threshold=request.threshold
        )
        
        return QueryResponse(
            answer=result["answer"],
            graph_context=result["graph_context"],
            chunk_context=result["chunk_context"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/graph", response_model=QueryResponse)
async def graph_search(
    request: QueryRequest,
    querying_service: GraphQuerying = Depends(get_querying_service)
):
    """Graph Search: Neo4j retrieval + LLM generation."""
    try:
        result = querying_service.graph_response(
            query=request.query,
            graph_limit=request.graph_limit
        )
        
        return QueryResponse(
            answer=result["answer"],
            graph_context=result["graph_context"],
            chunk_context=result["chunk_context"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
