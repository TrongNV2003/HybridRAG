import warnings
from loguru import logger
from langchain_community.document_loaders import WikipediaLoader

warnings.filterwarnings("ignore", message="No parser was explicitly specified")


class DataLoader:
    def __call__(self, query: str, load_max_docs: int = 10) -> list:
        return self.load_wikipedia(query, load_max_docs)
    
    
    def load_wikipedia(self, query: str, load_max_docs: int = 10) -> list:
        """
        Load documents from Wikipedia based on the query.
        
        Args:
            query (str): The query to search in Wikipedia.
            load_max_docs (int): Maximum number of documents to load.
        Returns:
            list: A list of documents with their content.
        """
        
        if query is not None:
            logger.info(f"Searching Wikipedia for: {query}")
            
            loader = WikipediaLoader(query=query, load_max_docs=load_max_docs)
            try:
                documents = loader.load()
            except Exception as e:
                logger.warning(f"Wikipedia loading failed for '{query}': {e}")
                return []

            docs = [
                {
                    "content": doc.page_content,
                    "metadata": {
                        "reference": "Wikipedia",
                        "title": doc.metadata.get("title", query),
                        "id": i
                    }
                }
                for i, doc in enumerate(documents)
            ]
            
            return docs
