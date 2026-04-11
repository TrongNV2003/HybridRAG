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

    def execute_query(self, query: str, auto_sync: bool = False) -> Tuple[List[Dict[str, Any]], List[str], bool]:
        """
        Execute a SPARQL query using rdflib on a synchronized RDF snapshot.
        
        Args:
            query (str): The SPARQL query string.
            auto_sync (bool): Whether to sync the static RDF file first.
            
        Returns:
            Tuple: (results, variables, sync_completed)
        """
        sync_completed = False
        if auto_sync:
            sync_completed = self.sync_rdf()
        
        logger.info("Executing SPARQL query via rdflib on local RDF snapshot...")
        
        try:
            # Parse/Refresh the RDF graph from the synchronized file
            if os.path.exists(self.rdf_file_path):
                self.rdf_graph = Graph()
                self.rdf_graph.parse(self.rdf_file_path, format="turtle")
            else:
                logger.warning(f"RDF file {self.rdf_file_path} not found. Attempting a fresh sync...")
                self.sync_rdf()
                if os.path.exists(self.rdf_file_path):
                    self.rdf_graph.parse(self.rdf_file_path, format="turtle")
                else:
                    return [], [], sync_completed
            
            # Execute query using rdflib's SPARQL engine
            query_res = self.rdf_graph.query(query)
            
            # Extract variables (column names)
            variables = [str(var) for var in query_res.vars]
            
            # Format results into list of dictionaries
            results = []
            for row in query_res:
                formatted_row = {}
                for var in variables:
                    # Accessing attribute by variable name for rdflib ResultRow
                    val = getattr(row, var)
                    formatted_row[var] = str(val) if val is not None else None
                results.append(formatted_row)
                
            return results, variables, sync_completed
            
        except Exception as e:
            logger.error(f"SPARQL query via rdflib failed: {e}")
            raise e