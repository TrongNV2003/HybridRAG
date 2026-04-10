from fastapi import APIRouter, Depends, HTTPException
from src.services.sparql_service import SparqlService
from src.config.schemas import SparqlRequest, SparqlResponse
from src.api.dependencies import get_sparql_service

router = APIRouter()

@router.post("/query", response_model=SparqlResponse)
async def sparql_query(
    request: SparqlRequest,
    sparql_service: SparqlService = Depends(get_sparql_service)
):
    """
    Execute a SPARQL query on the RDF data. 
    Automatically syncs from Neo4j if auto_sync is True.
    """
    try:
        results, variables, sync_completed = sparql_service.execute_query(
            query=request.query,
            auto_sync=request.auto_sync
        )
        
        return SparqlResponse(
            results=results,
            variables=variables,
            sync_completed=sync_completed,
            message="Query executed successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sync-rdf")
async def sync_rdf(
    sparql_service: SparqlService = Depends(get_sparql_service)
):
    """
    Manually synchronize Neo4j data to the RDF file.
    """
    success = sparql_service.sync_rdf()
    if success:
        return {"status": "success", "message": "RDF synchronization completed successfully"}
    else:
        raise HTTPException(status_code=500, detail="RDF synchronization failed")
