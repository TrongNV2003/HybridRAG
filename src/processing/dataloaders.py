import requests
import warnings
from loguru import logger
from langchain_community.document_loaders import WikipediaLoader

warnings.filterwarnings("ignore", message="No parser was explicitly specified")


class DataLoader:
    def __call__(self, query: str, load_max_docs: int = 10) -> list:
        return self.load_wikipedia(query, load_max_docs)
    
    def get_wikipedia_en_title(self, vi_title: str) -> str:
        """
        Get the corresponding English Wikipedia title for a Vietnamese title.
        
        Args:
            vi_title (str): The title of the Vietnamese Wikipedia page.
        Returns:
            str: The title of the English Wikipedia page, or None if not found.
        """
        url = "https://vi.wikipedia.org/w/api.php"
        headers = {
            "User-Agent": "HybridRAG-DBPedia-Version/1.0 (https://github.com/TrongNV2003/HybridRAG)"
        }
        params = {
            "action": "query",
            "prop": "langlinks",
            "lllang": "en",
            "titles": vi_title,
            "format": "json"
        }
        try:
            response = requests.get(url, params=params, headers=headers).json()
            pages = response.get("query", {}).get("pages", {})
            for page_id in pages:
                langlinks = pages[page_id].get("langlinks", [])
                if langlinks:
                    return langlinks[0].get("*")
        except Exception as e:
            logger.warning(f"Failed to fetch English title for '{vi_title}': {e}")
            
        return None
    
    
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
            
            loader = WikipediaLoader(
                query=query,
                lang="vi",
                load_max_docs=load_max_docs,
                doc_content_chars_max=5000
            )
            try:
                documents = loader.load()
            except Exception as e:
                logger.warning(f"Wikipedia loading failed for '{query}': {e}")
                return []

            docs = []
            for i, doc in enumerate(documents):
                vi_title = doc.metadata.get("title", query)
                en_title = self.get_wikipedia_en_title(vi_title)
                
                # Create full DBPedia URL for Standard 5*
                en_dbpedia_url = f"http://dbpedia.org/resource/{en_title.replace(' ', '_')}" if en_title else None
                
                docs.append({
                    "content": doc.page_content,
                    "metadata": {
                        "id": vi_title.strip().replace(" ", "_"),
                        "title": vi_title,
                        "en_dbpedia_url": en_dbpedia_url,
                        "source": doc.metadata.get("source", ""),
                        "summary": doc.metadata.get("summary", ""),
                        "reference": "Wikipedia"
                    }
                })
            
            return docs

if __name__ == "__main__":
    loader = DataLoader()
    docs = loader.load_wikipedia("Hồ Chí Minh", load_max_docs=1)
    print(docs)