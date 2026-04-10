import sys
from rdflib import Graph
from loguru import logger

def run_sparql_query(rdf_file, query_str):
    """
    Load an RDF file and execute a SPARQL query.
    """
    g = Graph()
    logger.info(f"Loading RDF file: {rdf_file}...")
    try:
        g.parse(rdf_file, format="turtle")
        logger.info(f"File loaded. Executing SPARQL query...")
        
        results = g.query(query_str)
        
        print(f"\n--- Query Results ({len(results)}) ---")
        for row in results:
            print(" | ".join([str(item) for item in row]))
        print("------------------------------\n")
        
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    file_path = "exports/test_vn_dbpedia.ttl"
    
    query = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX dbo: <http://dbpedia.org/ontology/>
    
    SELECT ?entity ?label ?type
    WHERE {
      ?entity rdfs:label ?label .
      ?entity rdf:type ?type .
    }
    LIMIT 10
    """
        
    run_sparql_query(file_path, query)
