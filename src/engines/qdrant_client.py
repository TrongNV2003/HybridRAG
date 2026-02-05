import logging
from typing import List, Dict, Any, Optional

from src.config.dataclass import VectorPoint
from src.engines.qdrant_search import QdrantSearchMixin
from src.config.settings import qdrant_config

logger = logging.getLogger(__name__)


class QdrantVectorStore(QdrantSearchMixin):
    """
    Qdrant vector store implementation.
    Supports both cloud and local Qdrant instances with hybrid search.
    """
    
    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        host: str = "localhost",
        port: int = 6333
    ):
        """
        Initialize Qdrant client.
        
        Args:
            url: Qdrant Cloud URL
            api_key: Qdrant API key
            host: Local Qdrant host
            port: Local Qdrant port
        """
        self._url = url or qdrant_config.url
        self._api_key = api_key or qdrant_config.api_key
        self._host = host or qdrant_config.host
        self._port = port or qdrant_config.port
        self._client = None
    
    def _get_client(self):
        """Get or create sync Qdrant client"""
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
                
                if self._url:
                    self._client = QdrantClient(
                        url=self._url,
                        api_key=self._api_key
                    )
                else:
                    self._client = QdrantClient(
                        host=self._host,
                        port=self._port
                    )
            except ImportError:
                raise ImportError("qdrant-client required. Install with: pip install qdrant-client")
        
        return self._client
    
    def create_collection(
        self,
        name: str,
        dimension: int,
        distance: str = "cosine",
        enable_sparse: bool = True
    ) -> bool:
        """
        Create a new collection in Qdrant with named vectors.
        
        Args:
            name: Collection name
            dimension: Dense vector dimension
            distance: Distance metric (cosine, euclidean, dot)
            enable_sparse: Enable sparse vectors for hybrid search
        """
        from qdrant_client.models import Distance, VectorParams, SparseVectorParams
        
        distance_map = {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "dot": Distance.DOT
        }
        
        client = self._get_client()
        
        try:
            # Named vectors config for hybrid search
            vectors_config = {
                "dense": VectorParams(
                    size=dimension,
                    distance=distance_map.get(distance, Distance.COSINE)
                )
            }
            
            # Sparse vectors config for keyword search
            sparse_vectors_config = None
            if enable_sparse:
                sparse_vectors_config = {
                    "sparse": SparseVectorParams()
                }
            
            client.create_collection(
                collection_name=name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config
            )
            logger.info(f"Created collection '{name}' with dimension {dimension}, sparse={enable_sparse}")
            return True
        except Exception as e:
            if "already exists" in str(e).lower():
                return True
            logger.error(f"Failed to create collection '{name}': {e}")
            raise
    
    def delete_collection(self, name: str) -> bool:
        """Delete a collection"""
        client = self._get_client()
        
        try:
            client.delete_collection(collection_name=name)
            logger.info(f"Deleted collection '{name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection '{name}': {e}")
            return False
    
    def collection_exists(self, name: str) -> bool:
        """Check if collection exists"""
        client = self._get_client()
        
        try:
            collections = client.get_collections()
            return any(c.name == name for c in collections.collections)
        except Exception as e:
            logger.error(f"Failed to check collection '{name}': {e}")
            return False
    
    def upsert(
        self,
        collection: str,
        points: List[VectorPoint]
    ) -> bool:
        """
        Insert or update vectors in Qdrant.
        Supports both dense-only and hybrid (dense + sparse) vectors.
        """
        from qdrant_client.models import PointStruct, SparseVector
        
        client = self._get_client()
        
        qdrant_points = []
        for p in points:
            vectors = {
                "dense": p.vector
            }
            
            # Add sparse vector if provided
            if p.sparse_indices and p.sparse_values:
                vectors["sparse"] = SparseVector(
                    indices=p.sparse_indices,
                    values=p.sparse_values
                )
            
            qdrant_points.append(PointStruct(
                id=p.id,
                vector=vectors,
                payload=p.payload
            ))
        
        try:
            client.upsert(
                collection_name=collection,
                points=qdrant_points
            )
            logger.debug(f"Upserted {len(points)} points to '{collection}'")
            return True
        except Exception as e:
            logger.error(f"Failed to upsert to '{collection}': {e}")
            raise
    
    def delete(
        self,
        collection: str,
        ids: List[str]
    ) -> bool:
        """Delete vectors by ID"""
        from qdrant_client.models import PointIdsList
        
        client = self._get_client()
        
        try:
            client.delete(
                collection_name=collection,
                points_selector=PointIdsList(points=ids)
            )
            logger.debug(f"Deleted {len(ids)} points from '{collection}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete from '{collection}': {e}")
            return False
    
    def delete_by_filter(
        self,
        collection: str,
        filter_conditions: Dict[str, Any]
    ) -> int:
        """Delete vectors matching filter conditions"""
        from qdrant_client.models import FilterSelector
        
        client = self._get_client()
        query_filter = self._build_filter(filter_conditions)
        
        try:
            client.delete(
                collection_name=collection,
                points_selector=FilterSelector(filter=query_filter)
            )
            logger.debug(f"Deleted points from '{collection}' with filter")
            return 1
        except Exception as e:
            logger.error(f"Failed to delete by filter from '{collection}': {e}")
            return 0
    
    def _build_filter(self, conditions: Dict[str, Any]):
        """Build Qdrant filter from conditions dict"""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        must_conditions = []
        
        for field, value in conditions.items():
            must_conditions.append(
                FieldCondition(
                    key=field,
                    match=MatchValue(value=value)
                )
            )
        
        return Filter(must=must_conditions)
    
    def get_collection_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get collection information"""
        client = self._get_client()
        
        try:
            info = client.get_collection(collection_name=name)
            
            def get_attr(obj, attr, default=None):
                return getattr(obj, attr, default)
            
            return {
                "name": name,
                "vectors_count": get_attr(info, "vectors_count", 0),
                "points_count": get_attr(info, "points_count", 0),
                "status": get_attr(info, "status").value if get_attr(info, "status") else "unknown"
            }
        except Exception as e:
            logger.error(f"Failed to get collection info for '{name}': {e}")
            return None
