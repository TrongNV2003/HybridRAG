import logging
import numpy as np
from typing import List, Dict, Any, Optional

from src.config.dataclass import SearchResult

logger = logging.getLogger(__name__)


class QdrantSearchMixin:
    """Mixin class for Qdrant search functionalities."""
    
    def dense_search(
        self,
        collection: str,
        vector: List[float],
        top_k: int = 5,
        threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar vectors (dense search)"""
        client = self._get_client()
        
        # Build filter if provided
        query_filter = None
        if filter_conditions:
            query_filter = self._build_filter(filter_conditions)
        
        try:
            results = client.query_points(
                collection_name=collection,
                query=vector,
                using="dense",
                limit=top_k,
                query_filter=query_filter,
                score_threshold=threshold
            )
        except Exception as e:
            if "Not existing vector name" in str(e) or "dense" in str(e):
                # Fallback to unnamed vector (old collections)
                logger.debug(f"Collection '{collection}' uses unnamed vectors, falling back")
                results = client.query_points(
                    collection_name=collection,
                    query=vector,
                    limit=top_k,
                    query_filter=query_filter,
                    score_threshold=threshold
                )
            else:
                logger.error(f"Search failed in '{collection}': {e}")
                raise
        
        return [
            SearchResult(
                id=str(r.id),
                score=r.score,
                payload=r.payload or {}
            )
            for r in results.points
        ]

    def sparse_search(
        self,
        collection: str,
        sparse_indices: List[int],
        sparse_values: List[float],
        top_k: int = 5,
        threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar sparse vectors (keyword search)"""
        from qdrant_client.models import SparseVector
        
        client = self._get_client()
        
        # Build filter if provided
        query_filter = None
        if filter_conditions:
            query_filter = self._build_filter(filter_conditions)
            
        try:
            results = client.query_points(
                collection_name=collection,
                query=SparseVector(indices=sparse_indices, values=sparse_values),
                using="sparse",
                limit=top_k,
                query_filter=query_filter,
                score_threshold=threshold,
                with_payload=True
            )
            
            return [
                SearchResult(
                    id=str(r.id),
                    score=r.score,
                    payload=r.payload or {}
                )
                for r in results.points
            ]
        except Exception as e:
            logger.warning(f"Sparse search failed in '{collection}': {e}")
            return []

    def hybrid_search(
        self,
        collection: str,
        query_vector: List[float],
        sparse_indices: List[int],
        sparse_values: List[float],
        top_k: int = 5,
        threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Qdrant hybrid search using prefetch + RRF fusion.
        Combines dense (semantic) and sparse (SPLADE) search.
        
        Args:
            query_vector: Dense embedding vector
            sparse_indices: Sparse vector indices (from sparse encoder)
            sparse_values: Sparse vector values (from sparse encoder)
        """
        from qdrant_client.models import Prefetch, SparseVector, FusionQuery, Fusion
        
        client = self._get_client()
        
        # Build filter if provided
        query_filter = None
        if filter_conditions:
            query_filter = self._build_filter(filter_conditions)
        
        try:
            prefetch_limit = top_k * 2
            
            results = client.query_points(
                collection_name=collection,
                prefetch=[
                    Prefetch(
                        query=query_vector,
                        using="dense",
                        limit=prefetch_limit
                    ),
                    Prefetch(
                        query=SparseVector(indices=sparse_indices, values=sparse_values),
                        using="sparse",
                        limit=prefetch_limit
                    )
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
                score_threshold=threshold
            )
            
            search_results = [
                SearchResult(
                    id=str(r.id),
                    score=r.score,
                    payload=r.payload or {}
                )
                for r in results.points
            ]
            
            logger.debug(f"Hybrid search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.warning(f"Server-side hybrid search failed: {e}. Falling back to client-side RRF.")
            
            try:
                dense_results = self.dense_search(
                    collection=collection,
                    vector=query_vector,
                    top_k=top_k * 2,
                    threshold=threshold,
                    filter_conditions=filter_conditions
                )
                
                sparse_results = self.sparse_search(
                    collection=collection,
                    sparse_indices=sparse_indices,
                    sparse_values=sparse_values,
                    top_k=top_k * 2,
                    threshold=threshold,
                    filter_conditions=filter_conditions
                )

                if not sparse_results:
                    return dense_results[:top_k]
                
                fused_results = self._compute_rrf(dense_results, sparse_results)
                return fused_results[:top_k]
                
            except Exception as final_e:
                logger.error(f"Client-side fallback also failed: {final_e}. Returning empty.")
                return []


    def _normalize_scores(self, scores: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """Normalize scores to [0,1] range using different methods."""
        if len(scores) == 0:
            return scores
            
        if method == 'softmax':
            # Softmax normalization
            exp_scores = np.exp(scores - np.max(scores))  # Subtract max for numerical stability
            return exp_scores / np.sum(exp_scores)
        else:  # minmax
            # Min-max normalization
            score_min = np.min(scores)
            score_max = np.max(scores)
            if score_max == score_min:
                return np.ones_like(scores)
            return (scores - score_min) / (score_max - score_min)

    def _compute_rrf(
        self, 
        dense_results: List[SearchResult], 
        sparse_results: List[SearchResult], 
        dense_weight: float = 0.5, 
        sparse_weight: float = 0.5, 
        rrf_k: float = 60.0
    ) -> List[SearchResult]:
        """Compute RRF scores for dense and sparse results."""
        if not dense_results and not sparse_results:
            return []
            
        dense_scores_map = {r.id: r.score for r in dense_results}
        sparse_scores_map = {r.id: r.score for r in sparse_results}
        
        all_ids = list(set(dense_scores_map.keys()) | set(sparse_scores_map.keys()))
        
        if not all_ids:
            return []

        d_scores = np.array([dense_scores_map.get(uid, 0.0) for uid in all_ids])
        s_scores = np.array([sparse_scores_map.get(uid, 0.0) for uid in all_ids])
        
        d_norm = self._normalize_scores(d_scores, 'softmax')
        s_norm = self._normalize_scores(s_scores, 'softmax')
        
        d_ranks = np.argsort(-d_norm).argsort() + 1
        s_ranks = np.argsort(-s_norm).argsort() + 1
        
        final_scores = []
        payload_map = {r.id: r.payload for r in dense_results + sparse_results}
        
        for i, uid in enumerate(all_ids):
            score = 0.0
            if uid in dense_scores_map:
                score += dense_weight / (rrf_k + d_ranks[i])
            
            if uid in sparse_scores_map:
                score += sparse_weight / (rrf_k + s_ranks[i])
            
            final_scores.append(SearchResult(
                id=uid,
                score=float(score),
                payload=payload_map.get(uid, {})
            ))
            
        final_scores.sort(key=lambda x: x.score, reverse=True)
        return final_scores
