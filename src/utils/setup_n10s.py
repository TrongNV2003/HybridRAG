from langchain_neo4j import Neo4jGraph
from loguru import logger
from src.config.settings import neo4j_config

def setup_neosemantics():
    """
    Initialize and configure Neosemantics (n10s) on the connected Neo4j instance.
    Sets up namespaces and property mappings.
    """
    logger.info("Connecting to Neo4j for Neosemantics setup...")
    try:
        graph_db = Neo4jGraph(
            url=neo4j_config.url,
            username=neo4j_config.username,
            password=neo4j_config.password
        )
        
        # 0. Check if n10s is installed
        check_proc = "CALL dbms.procedures() YIELD name WHERE name STARTS WITH 'n10s.graphconfig.init' RETURN count(*) as n"
        res = graph_db.query(check_proc)
        if res[0]['n'] == 0:
            logger.error("Neosemantics plugin NOT found. Please restart Docker after updating docker-compose.yml.")
            return

        # 1. Initialize Graph Config (if not already initialized)
        # Using MAP handles vocabularies without needing to store full URIs in nodes
        logger.info("Initializing n10s GraphConfig...")
        graph_db.query("CALL n10s.graphconfig.init({ handleVocabUris: 'MAP' })")
        
        # 2. Add Constraint (Required by n10s)
        logger.info("Creating uniqueness constraint for n10s Resource...")
        graph_db.query("CREATE CONSTRAINT n10s_unique_uri IF NOT EXISTS FOR (r:Resource) REQUIRE r.uri IS UNIQUE")

        # 3. Define Namespaces
        logger.info("Defining RDF Namespaces...")
        namespaces = {
            "dbo": "http://dbpedia.org/ontology/",
            "dbr-vi": "http://vi.dbpedia.org/resource/",
            "dbr-en": "http://dbpedia.org/resource/",
            "owl": "http://www.w3.org/2002/07/owl#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#"
        }
        for prefix, uri in namespaces.items():
            graph_db.query(f"CALL n10s.nsprefixes.add('{prefix}', '{uri}')")

        # 4. Define Mappings (Property Graph -> RDF)
        logger.info("Defining Semantic Mappings...")
        
        # Map Node labels
        graph_db.query("CALL n10s.mapping.add('http://dbpedia.org/ontology/Thing', 'Entity')")
        
        # Map Properties
        graph_db.query("CALL n10s.mapping.add('http://www.w3.org/2000/01/rdf-schema#label', 'id')")
        graph_db.query("CALL n10s.mapping.add('http://www.w3.org/2000/01/rdf-schema#comment', 'entity_role')")
        graph_db.query("CALL n10s.mapping.add('http://www.w3.org/2002/07/owl#sameAs', 'en_dbpedia_url')")

        logger.info("Neosemantics setup COMPLETED successfully.")

    except Exception as e:
        logger.error(f"Failed to setup Neosemantics: {e}")

if __name__ == "__main__":
    setup_neosemantics()
