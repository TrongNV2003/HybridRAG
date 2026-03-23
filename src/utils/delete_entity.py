from loguru import logger
from src.api.dependencies import get_neo4j_graph, get_qdrant_store
from src.config.settings import qdrant_config

def delete_entity_flower(entity_name: str):
    logger.info(f"Đang tìm và xóa thực thể trung tâm chứa từ khóa: '{entity_name}' và các node liên quan...")
    graph = get_neo4j_graph()
    
    # Tìm thực thể trung tâm
    query_center = """
    MATCH (c:Entity)
    WHERE toLower(c.id) CONTAINS toLower($entity_name)
    RETURN c.id as entity_id, c.reference as reference
    """
    centers = graph.query(query_center, params={"entity_name": entity_name})
    if not centers:
        logger.warning(f"Không tìm thấy thực thể nào có tên chứa '{entity_name}'")
        return
        
    for center in centers:
        c_id = center['entity_id']
        logger.info(f"Đã tìm thấy thực thể trung tâm: '{c_id}'")
        
        # Xóa các node "lá" (leaf nodes) chỉ liên kết duy nhất với thực thể trung tâm này
        query_leaves = """
        MATCH (leaf:Entity)-[r]-(c:Entity {id: $c_id})
        WITH c, leaf, count { (leaf)--() } as degree
        WHERE degree = 1
        DETACH DELETE leaf
        RETURN count(leaf) as deleted_leaves
        """
        leaves_res = graph.query(query_leaves, params={"c_id": c_id})
        num_leaves = leaves_res[0]['deleted_leaves'] if leaves_res else 0
        if num_leaves > 0:
            logger.info(f"Đoá hoa: Đã xóa {num_leaves} node 'lá' chỉ liên kết với '{c_id}'.")
            
        # Xóa bản thân thực thể trung tâm
        query_delete_center = """
        MATCH (c:Entity {id: $c_id})
        DETACH DELETE c
        """
        graph.query(query_delete_center, params={"c_id": c_id})
        logger.info(f"Đã xóa thực thể trung tâm: '{c_id}' và các cạnh kết nối nó.")
        
    # Tìm và dọn dẹp các Chunk "mồ côi" (không còn liên kết với Entity nào)
    query_clean_chunks = """
    MATCH (c:Chunk)
    WHERE count { (c)-[:MENTIONS]->() } = 0
    WITH c.id as chunk_id, c
    DETACH DELETE c
    RETURN chunk_id
    """
    clean_res = graph.query(query_clean_chunks)
    chunk_ids = [res['chunk_id'] for res in (clean_res or [])]
    
    if chunk_ids:
        logger.info(f"Đã dọn dẹp {len(chunk_ids)} Chunk mồ côi (không còn entity nào để tham chiếu) khỏi Neo4j.")
        
        # Xóa các embedding tương ứng của các Chunk mồ côi khỏi Qdrant
        try:
            qdrant_store = get_qdrant_store()
            qdrant_store.client.delete(
                collection_name=qdrant_config.collection_name,
                points_selector=chunk_ids
            )
            logger.info(f"Đã xóa {len(chunk_ids)} vector embeddings tương ứng khỏi Qdrant.")
        except Exception as e:
            logger.error(f"Lỗi khi xóa embeddings mồ côi từ Qdrant: {e}")
    else:
        logger.info("Không có Chunk mồ côi nào cần dọn dẹp.")

if __name__ == "__main__":
    entity_name = 'Richard Ira "Dick" Bong'
    delete_entity_flower(entity_name)
