import os
from rdflib import Graph, Literal, RDF, URIRef, Namespace
from rdflib.namespace import OWL, RDFS
from langchain_neo4j import Neo4jGraph
from loguru import logger
from src.config.settings import neo4j_config

# Define Namespaces
DBO = Namespace("http://dbpedia.org/ontology/")
DBR_VI = Namespace("http://vi.dbpedia.org/resource/")
DBR_EN = Namespace("http://dbpedia.org/resource/")

# Relationship mapping from Vietnamese/LLM-extracted terms to standard DBO predicates
RELATIONSHIP_MAP = {
    "sinh_tại": "birthPlace",
    "mất_tại": "deathPlace",
    "cha_của": "parent",
    "mẹ_của": "parent",
    "vợ_của": "spouse",
    "chồng_của": "spouse",
    "con_của": "child",
    "anh_của": "relative",
    "chị_của": "relative",
    "em_của": "relative",
    "là_tác_giả_của": "author",
    "là_thành_viên_của": "member",
    "thuộc_quốc_gia": "country",
    "thuộc_tỉnh": "province",
    "nằm_trong": "isPartOf",
    "là_thủ_đô_của": "capital",
    "là_tên_khác_của": "alias",
    "lãnh_đạo_của": "commander",
}

# Entity type mapping for standardizing DBO classes
ENTITY_TYPE_MAP = {
    "người": "Person",
    "địa_điểm": "Place",
    "tổ_chức": "Organisation",
    "sự_kiện": "Event",
    "tác_phẩm": "Work",
    "quốc_gia": "Country",
    "tỉnh": "Province",
    "thành_phố": "City",
}

class RDFExporter:
    """
    RDFExporter handles converting Neo4j Property Graph data into RDF format.
    Standardizes on DBPedia ontology (4* and 5*).
    """
    def __init__(self, graph_db: Neo4jGraph):
        self.graph_db = graph_db
        self.rdf_graph = Graph()
        self.rdf_graph.bind("dbo", DBO)
        self.rdf_graph.bind("dbr-vi", DBR_VI)
        self.rdf_graph.bind("dbr-en", DBR_EN)
        self.rdf_graph.bind("owl", OWL)

    def fetch_and_convert(self):
        """Fetch all data from Neo4j and convert to RDF triples."""
        logger.info("Fetching data from Neo4j for RDF conversion...")
        
        # 1. Convert Nodes (Entities)
        node_query = "MATCH (n:Entity) RETURN n"
        nodes = self.graph_db.query(node_query)
        
        for record in nodes:
            node = record['n']
            node_id = node.get('id', '').replace(" ", "_")
            if not node_id:
                continue
                
            subject_uri = DBR_VI[node_id]
            
            # rdf:type
            raw_type = node.get('entity_type', 'Thing')
            entity_type = ENTITY_TYPE_MAP.get(raw_type.lower(), raw_type)
            self.rdf_graph.add((subject_uri, RDF.type, DBO[entity_type]))
            
            # rdfs:label
            self.rdf_graph.add((subject_uri, RDFS.label, Literal(node.get('id', ''), lang="vi")))
            
            # rdfs:comment (using entity_role as description)
            role = node.get('entity_role')
            if role:
                self.rdf_graph.add((subject_uri, RDFS.comment, Literal(role, lang="vi")))
            
            # owl:sameAs (Standard 5* - Interlinking with English DBPedia)
            en_url = node.get('en_dbpedia_url')
            if en_url:
                en_uri = URIRef(en_url)
                self.rdf_graph.add((subject_uri, OWL.sameAs, en_uri))

        # 2. Convert Relationships
        rel_query = "MATCH (s:Entity)-[r]->(t:Entity) RETURN s.id as source, type(r) as type, t.id as target"
        rels = self.graph_db.query(rel_query)
        
        for rel in rels:
            source_uri = DBR_VI[rel['source'].replace(" ", "_")]
            target_uri = DBR_VI[rel['target'].replace(" ", "_")]
            
            raw_type = rel['type']
            # Map Vietnamese/Raw type to standard English DBO property if exists
            mapped_type = RELATIONSHIP_MAP.get(raw_type.lower(), raw_type)
            predicate_uri = DBO[mapped_type]
            
            self.rdf_graph.add((source_uri, predicate_uri, target_uri))

        logger.info(f"Converted {len(self.rdf_graph)} triples.")

    def export(self, output_path: str, format: str = "turtle"):
        """Export the RDF graph to a file."""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            self.rdf_graph.serialize(destination=output_path, format=format)
            logger.info(f"RDF data exported to {output_path} in {format} format.")
        except Exception as e:
            logger.error(f"Failed to export RDF: {e}")

if __name__ == "__main__":
    # Test export
    graph_db = Neo4jGraph(
        url=neo4j_config.url,
        username=neo4j_config.username,
        password=neo4j_config.password
    )
    exporter = RDFExporter(graph_db)
    exporter.fetch_and_convert()
    exporter.export("exports/vietnamese_dbpedia.ttl")
