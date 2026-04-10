import os
from rdflib import Graph
from loguru import logger
from typing import List, Dict, Any, Tuple
from langchain_neo4j import Neo4jGraph

from src.utils.rdf_exporter import RDFExporter

class SparqlService:
    """
    SparqlService manages RDF synchronization from Neo4j and 
    executes SPARQL queries on the generated RDF data.
    """
    def __init__(self, graph_db: Neo4jGraph, rdf_file_path: str = "exports/vietnamese_dbpedia.ttl"):
        self.graph_db = graph_db
        self.rdf_file_path = rdf_file_path
        self.rdf_graph = Graph()

    def sync_rdf(self) -> bool:
        """
        Synchronize Neo4j data to RDF file.
        Returns:
            bool: True if successful, False otherwise.
        """
        logger.info("Starting RDF synchronization from Neo4j...")
        try:
            exporter = RDFExporter(self.graph_db)
            exporter.fetch_and_convert()
            exporter.export(self.rdf_file_path)
            
            # Log completion as requested by user
            logger.info("RDF synchronization COMPLETED successfully.")
            return True
        except Exception as e:
            logger.error(f"RDF synchronization FAILED: {e}")
            return False

    def execute_query(self, query: str, auto_sync: bool = True) -> Tuple[List[Dict[str, Any]], List[str], bool]:
        """
        Execute a SPARQL query using Neosemantics (n10s) directly on Neo4j.
        
        Args:
            query (str): The SPARQL query string.
            auto_sync (bool): Whether to sync the static RDF file (for export purposes).
            
        Returns:
            Tuple: (results, variables, sync_completed)
        """
        sync_completed = False
        if auto_sync:
            sync_completed = self.sync_rdf()
        
        logger.info("Executing SPARQL query via Neosemantics (n10s) natively on Neo4j...")
        
        # Use n10s.sparql.query procedure in Neo4j
        # This executes the SPARQL query directly on the property graph using mappings
        cypher_query = "CALL n10s.sparql.query($sparql) YIELD keys, row"
        
        try:
            records = self.graph_db.query(cypher_query, {"sparql": query})
            
            if not records:
                return [], [], sync_completed
                
            variables = records[0].get("keys", [])
            results = []
            
            for record in records:
                row_data = record.get("row", {})
                # Format each row to match SparqlResponse schema
                formatted_row = {var: str(row_data.get(var)) if row_data.get(var) is not None else None for var in variables}
                results.append(formatted_row)
                
            return results, variables, sync_completed
            
        except Exception as e:
            logger.error(f"SPARQL query via n10s failed: {e}")
            raise e
