import os
from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from data_manager import DataManager

class VectorStore:
    """
    A class for managing Dota 2 data in a Chroma vector database using LangChain.
    Stores hero information, matchups, and role data for semantic search.
    """

    def __init__(self):
        """Initialize the VectorStore with Chroma and data manager."""
        self.db_path = "chroma_langchain_db"
        self.data_manager = DataManager()

        # Initialize embedding model
        self.embeddings = OllamaEmbeddings(
            model="qwen3:0.6b",
            base_url="http://localhost:11434"
        )

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        # Initialize or load existing stores
        self._initialize_stores()

    def _initialize_stores(self):
        """Initialize or load existing vector stores."""
        hero_path = os.path.join(self.db_path, "heroes")
        matchup_path = os.path.join(self.db_path, "matchups")

        if os.path.exists(hero_path) and os.path.exists(matchup_path):
            print("Loading existing vector stores...")
            # Load existing stores
            self.hero_store = Chroma(
                persist_directory=hero_path,
                embedding_function=self.embeddings
            )
            self.matchup_store = Chroma(
                persist_directory=matchup_path,
                embedding_function=self.embeddings
            )
        else:
            print("Creating new vector stores...")
            # Create new stores
            self.load_data()

    def _filter_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter metadata to ensure all values are of types that Chroma can handle.

        Args:
            metadata (Dict[str, Any]): Original metadata dictionary

        Returns:
            Dict[str, Any]: Filtered metadata dictionary
        """
        filtered = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                filtered[key] = value
            elif isinstance(value, list):
                filtered[key] = ", ".join(str(item) for item in value)
            else:
                filtered[key] = str(value)
        return filtered

    def _create_hero_documents(self) -> List[Document]:
        """
        Create Document objects for hero data.

        Returns:
            List[Document]: List of Document objects containing hero information
        """
        hero_directory = self.data_manager.get_hero_directory()
        documents = []

        for hero in hero_directory:
            roles = [role["roleId"] for role in hero.get("roles", [])]
            content = f"""
            Hero: {hero['displayName']}
            Short Name: {hero['shortName']}
            Roles: {', '.join(roles)}
            """

            # Create document with filtered metadata
            metadata = {
                "id": str(hero["id"]),
                "name": hero["displayName"],
                "short_name": hero["shortName"],
                "roles": roles,
                "type": "hero"
            }

            filtered_metadata = self._filter_metadata(metadata)

            doc = Document(
                page_content=content,
                metadata=filtered_metadata
            )
            documents.append(doc)

        return documents

    def _create_matchup_documents(self) -> List[Document]:
        """
        Create Document objects for matchup data.

        Returns:
            List[Document]: List of Document objects containing matchup information
        """
        matchups = self.data_manager.get_matchups()
        documents = []

        for hero_id, matchup_data in matchups.items():
            advantages = []
            disadvantages = []

            for section in ['advantage', 'disadvantage']:
                for group in matchup_data.get(section, []):
                    for kind in ['with', 'vs']:
                        for matchup in group.get(kind, []):
                            h1 = matchup.get('heroId1')
                            h2 = matchup.get('heroId2')
                            win_rate = matchup.get('winRateHeroId1', 0.5)
                            if h1 is not None and h2 is not None:
                                if section == 'advantage':
                                    advantages.append(f"Hero {h1} has {win_rate:.1%} win rate against/with Hero {h2}")
                                else:
                                    disadvantages.append(f"Hero {h1} has {win_rate:.1%} win rate against/with Hero {h2}")

            content = f"""
            Hero ID: {hero_id}
            Advantages:
            {chr(10).join(advantages)}

            Disadvantages:
            {chr(10).join(disadvantages)}
            """

            # Create document with filtered metadata
            metadata = {
                "id": f"matchup_{hero_id}",
                "hero_id": hero_id,
                "type": "matchup"
            }

            filtered_metadata = self._filter_metadata(metadata)

            doc = Document(
                page_content=content,
                metadata=filtered_metadata
            )
            documents.append(doc)

        return documents

    def load_data(self):
        """Load all Dota 2 data into the vector database."""
        print("Loading hero data...")
        # Create documents
        hero_docs = self._create_hero_documents()
        matchup_docs = self._create_matchup_documents()

        print("Splitting documents into chunks...")
        # Split documents into chunks
        hero_splits = self.text_splitter.split_documents(hero_docs)
        matchup_splits = self.text_splitter.split_documents(matchup_docs)

        print("Creating vector stores...")
        # Create vector stores
        self.hero_store = Chroma.from_documents(
            documents=hero_splits,
            embedding=self.embeddings,
            persist_directory=os.path.join(self.db_path, "heroes")
        )

        self.matchup_store = Chroma.from_documents(
            documents=matchup_splits,
            embedding=self.embeddings,
            persist_directory=os.path.join(self.db_path, "matchups")
        )

        # Persist the stores to disk
        print("Persisting vector stores to disk...")
        self.hero_store.persist()
        self.matchup_store.persist()
        print("Vector stores created and persisted successfully!")

    def search_heroes(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for heroes based on a query.

        Args:
            query (str): Search query
            n_results (int): Number of results to return

        Returns:
            List[Dict[str, Any]]: List of matching heroes with their metadata
        """
        results = self.hero_store.similarity_search_with_score(query, k=n_results)
        return [
            {
                "id": doc.metadata["id"],
                "metadata": {
                    "name": doc.metadata["name"],
                    "short_name": doc.metadata["short_name"],
                    "roles": doc.metadata["roles"]
                },
                "distance": score
            }
            for doc, score in results
        ]

    def search_matchups(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for matchups based on a query.

        Args:
            query (str): Search query
            n_results (int): Number of results to return

        Returns:
            List[Dict[str, Any]]: List of matching matchups with their metadata
        """
        results = self.matchup_store.similarity_search_with_score(query, k=n_results)
        return [
            {
                "id": doc.metadata["id"],
                "metadata": {
                    "hero_id": doc.metadata["hero_id"]
                },
                "distance": score
            }
            for doc, score in results
        ]

if __name__ == "__main__":
    # Example usage
    vector_store = VectorStore()
    vector_store.load_data()

    # Test searches
    print("\nSearching for carry heroes:")
    results = vector_store.search_heroes("heroes that are good at carrying")
    for result in results:
        print(f"{result['metadata']['name']}: {result['distance']}")

    print("\nSearching for strong matchups:")
    results = vector_store.search_matchups("heroes with high win rates")
    for result in results:
        print(f"Hero {result['metadata']['hero_id']}: {result['distance']}")