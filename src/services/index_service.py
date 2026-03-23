import uuid
from openai import OpenAI
from loguru import logger
from langchain_neo4j import Neo4jGraph
from typing import List, Optional, Dict

from src.config.dataclass import StructuralChunk
from src.engines.llm_engine import EntityExtractionLLM
from src.core.storage import GraphStorage, EmbedStorage
from src.processing.chunking import TwoPhaseDocumentChunker
from src.processing.postprocessing import EntityPostprocessor


class GraphIndexing:
    def __init__(
        self,
        client: OpenAI,
        graph_db: Neo4jGraph,
        chunker: TwoPhaseDocumentChunker,
        extractor: EntityExtractionLLM,
        postprocessor: EntityPostprocessor,
        storage: GraphStorage,
        qdrant_storage: EmbedStorage,
        clear_old_graph: bool = False
    ):
        self.client = client
        self.graph_db = graph_db
        self.chunker = chunker
        self.storage = storage
        self.extractor = extractor
        self.postprocessor = postprocessor
        self.qdrant_storage = qdrant_storage
        self.clear_old_graph = clear_old_graph

    def chunking(self, document: Dict, max_new_chunk_size: Optional[int] = None) -> List['StructuralChunk']:
        content = document.get("content", "")
        metadata = document.get("metadata", {})
        chunks = self.chunker.chunk_document(
            content, 
            max_new_chunk_size=max_new_chunk_size,
            additional_metadata=metadata
        )
        return chunks

    def indexing(self, chunks: List['StructuralChunk']) -> None:
        """Index pre-chunked data into the graph database."""
        batch_size = 5
        all_nodes: List[Dict] = []
        all_relationships: List[Dict] = []
        batch_nodes: List[Dict] = []
        batch_relationships: List[Dict] = []
        batch_chunk_relationships: List[Dict] = []
        batch_chunks: List['StructuralChunk'] = []
        total_chunks = len(chunks)
        
        if self.clear_old_graph:
            logger.info("Clearing existing graph data")
            self.storage.clear_all()
            self.qdrant_storage.clear_collection()

        for i, chunk in enumerate(chunks):
            text = getattr(chunk, "content", None) or chunk.get("content", "")
            if not text:
                continue

            logger.info(f"Processing chunk {i + 1}/{total_chunks}")
            
            try:
                extracted_data = self.extractor.call(text=text)
            except Exception:
                logger.warning(f"API call failed for chunk {i + 1}")
                continue
            
            if not extracted_data or "nodes" not in extracted_data or "relationships" not in extracted_data:
                continue
            
            cleaned_nodes: List[Dict] = []
            chunk_provenance: List[Dict] = []
            chunk_id = chunk.metadata.get("chunk_id")
            if not chunk_id:
                import hashlib
                ref_str = chunk.metadata.get("reference", "Unknown")
                hash_input = f"{ref_str}::{text}".encode("utf-8")
                chunk_id = str(uuid.UUID(hashlib.md5(hash_input).hexdigest()))
            # Ensure chunk has ID in metadata
            chunk.metadata["chunk_id"] = chunk_id

            for node in extracted_data.get("nodes", []):
                node_id = self.postprocessor(node.get("id", ""))
                cleaned_nodes.append({
                    "id": node_id,
                    "entity_type": self.postprocessor(node.get("entity_type", "")),
                    "entity_role": self.postprocessor(node.get("entity_role", "")),
                    "reference": chunk.metadata.get("reference", "Unknown")
                })
                # Link Chunk -> Entity
                chunk_provenance.append({
                    "source": chunk_id,
                    "target": node_id
                })
            
            cleaned_relationships: List[Dict] = []
            for rel in extracted_data.get("relationships", []):
                cleaned_relationships.append({
                    "source": self.postprocessor(rel.get("source", "")),
                    "target": self.postprocessor(rel.get("target", "")),
                    "relationship_type": self.postprocessor(rel.get("relationship_type", "")),
                })
            
            batch_chunk_relationships.extend(chunk_provenance)
            batch_nodes.extend(cleaned_nodes)
            batch_relationships.extend(cleaned_relationships)
            
            # Prepare chunks for embedding storage
            batch_chunks.append(chunk)
            
            logger.info(f"Extracted chunk {i + 1}: {len(cleaned_nodes)} entities, {len(cleaned_relationships)} relationships")

            if (i + 1) % batch_size == 0 or (i + 1) == total_chunks:
                dedup_nodes = self._deduplicate_entities(batch_nodes)
                dedup_relationships = self._deduplicate_relationships(batch_relationships)
                
                dedup_chunk_rels = [dict(t) for t in {tuple(d.items()) for d in batch_chunk_relationships}]

                batch_graph_data = {
                    "nodes": dedup_nodes,
                    "relationships": dedup_relationships,
                    "chunks": batch_chunks,
                    "chunk_relationships": dedup_chunk_rels
                }
                self.storage.store_graph(batch_graph_data)
                
                all_nodes.extend(dedup_nodes)
                all_relationships.extend(dedup_relationships)

                # Store embeddings to Qdrant for hybrid search
                if batch_chunks:
                    self.qdrant_storage.store_embeddings(batch_chunks)
                
                # Reset batch
                batch_nodes = []
                batch_relationships = []
                batch_chunks = []
                batch_chunk_relationships = []

        # Global deduplication across all batches to get accurate counts for the current session
        self._final_nodes = self._deduplicate_entities(all_nodes)
        self._final_rels = self._deduplicate_relationships(all_relationships)
        
        counts = {
            "entities_count": len(self._final_nodes),
            "relationships_count": len(self._final_rels),
            "chunks_count": total_chunks
        }
        
        logger.info(f"Indexing Session Summary - Entities: {counts['entities_count']}, Relationships: {counts['relationships_count']}, Chunks: {counts['chunks_count']}")
        
        return counts
            
    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """Remove duplicate entities based on (id, entity_type, entity_role)."""
        seen = set()
        deduplicated: List[Dict] = []
        
        for entity in entities:
            entity_id = entity.get('id', '').strip()
            entity_type = entity.get('entity_type', '').strip()
            entity_role = entity.get('entity_role', '').strip()
            
            if not entity_id:
                continue
            
            key = (entity_id, entity_type, entity_role)
            if key not in seen:
                seen.add(key)
                deduplicated.append(entity)
        
        return deduplicated

    def _deduplicate_relationships(self, relationships: List[Dict]) -> List[Dict]:
        """
        Remove duplicate relationships based on source, target, and relationship_type.
        If duplicates exist, keep the first occurrence.
        """
        seen = set()
        deduplicated = []
        
        for relationship in relationships:
            source = relationship.get('source', '')
            target = relationship.get('target', '')
            rel_type = relationship.get('relationship_type', '')
            
            key = (source, target, rel_type)
            
            if key not in seen:
                seen.add(key)
                deduplicated.append(relationship)
            else:
                logger.debug(f"Duplicate relationship found and removed: source='{source}', target='{target}', type='{rel_type}'")
        
        return deduplicated
