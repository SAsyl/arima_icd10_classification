#!/usr/bin/env python3
"""
Script for searching medical protocols in ChromaDB.
Provides functionality to query the vector database and retrieve relevant protocol chunks.
"""

import logging
import argparse
import sys
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from protocol_processor_fixed import ProtocolEmbeddingFunction

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProtocolSearcher:
    """Class for searching medical protocols in ChromaDB."""
    
    def __init__(
        self,
        chroma_persist_directory: str = "./chroma_db",
        collection_name: str = "medical_protocols",
        embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ):
        """
        Initialize the protocol searcher.
        
        Args:
            chroma_persist_directory: Directory where ChromaDB is persisted
            collection_name: Name of the ChromaDB collection
            embedding_model_name: Name of the embedding model
        """
        self.chroma_persist_directory = chroma_persist_directory
        self.collection_name = collection_name
        
        # Initialize embedding function
        self.embedding_function = ProtocolEmbeddingFunction(embedding_model_name)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=chroma_persist_directory)
        
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Connected to collection '{collection_name}'")
        except Exception as e:
            logger.error(f"Failed to connect to collection '{collection_name}': {e}")
            sys.exit(1)
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search for relevant protocol chunks.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            where: Metadata filter conditions
            where_document: Document content filter conditions
            
        Returns:
            Dictionary containing search results
        """
        logger.info(f"Searching for: '{query}' (top {n_results} results)")
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
                where_document=where_document
            )
            
            # Log results summary
            if results['ids'] and results['ids'][0]:
                logger.info(f"Found {len(results['ids'][0])} results")
            else:
                logger.info("No results found")
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {'ids': [[]], 'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
    
    def get_protocol_chunks(self, protocol_id: str) -> Dict[str, Any]:
        """
        Get all chunks for a specific protocol.
        
        Args:
            protocol_id: ID of the protocol to retrieve
            
        Returns:
            Dictionary containing protocol chunks
        """
        logger.info(f"Retrieving all chunks for protocol: {protocol_id}")
        
        try:
            results = self.collection.get(
                where={'protocol_id': protocol_id},
                include=['documents', 'metadatas', 'embeddings']
            )
            
            # Sort by chunk number
            if results['metadatas']:
                sorted_items = sorted(
                    zip(results['ids'], results['documents'], results['metadatas']),
                    key=lambda x: x[2]['chunk_number']
                )
                
                sorted_ids = [item[0] for item in sorted_items]
                sorted_documents = [item[1] for item in sorted_items]
                sorted_metadatas = [item[2] for item in sorted_items]
                
                results['ids'] = sorted_ids
                results['documents'] = sorted_documents
                results['metadatas'] = sorted_metadatas
            
            logger.info(f"Retrieved {len(results['ids'])} chunks for protocol {protocol_id}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve protocol {protocol_id}: {e}")
            return {'ids': [], 'documents': [], 'metadatas': []}
    
    def list_protocols(self) -> List[str]:
        """
        List all protocol IDs in the collection.
        
        Returns:
            List of protocol IDs
        """
        try:
            # Get all unique protocol IDs
            results = self.collection.get(include=['metadatas'])
            protocol_ids = set()
            
            for metadata in results['metadatas']:
                if metadata and 'protocol_id' in metadata:
                    protocol_ids.add(metadata['protocol_id'])
            
            protocol_list = sorted(list(protocol_ids))
            logger.info(f"Found {len(protocol_list)} protocols in collection")
            return protocol_list
            
        except Exception as e:
            logger.error(f"Failed to list protocols: {e}")
            return []
    
    def print_search_results(self, results: Dict[str, Any], query: str) -> None:
        """
        Print search results in a readable format.
        
        Args:
            results: Search results from the query method
            query: Original search query
        """
        print(f"\n{'='*80}")
        print(f"SEARCH RESULTS FOR: '{query}'")
        print(f"{'='*80}")
        
        if not results['ids'] or not results['ids'][0]:
            print("No results found.")
            return
        
        for i, (doc_id, document, metadata, distance) in enumerate(zip(
            results['ids'][0],
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            print(f"\n{'-'*60}")
            print(f"RESULT {i+1} (Similarity: {1-distance:.4f})")
            print(f"{'-'*60}")
            print(f"Document ID: {doc_id}")
            print(f"Protocol ID: {metadata.get('protocol_id', 'N/A')}")
            print(f"Source File: {metadata.get('source_file', 'N/A')}")
            print(f"Title: {metadata.get('title', 'N/A')}")
            print(f"Chunk: {metadata.get('chunk_number', 'N/A')} of {metadata.get('total_chunks', 'N/A')}")
            print(f"ICD Codes: {metadata.get('icd_codes_str', 'N/A')}")
            print(f"\nContent:")
            # print(document[:500] + "..." if len(document) > 500 else document)
            print(len(document), document)
    
    def print_protocol_chunks(self, protocol_id: str, results: Dict[str, Any]) -> None:
        """
        Print protocol chunks in a readable format.
        
        Args:
            protocol_id: Protocol ID
            results: Results from get_protocol_chunks method
        """
        print(f"\n{'='*80}")
        print(f"PROTOCOL: {protocol_id}")
        print(f"{'='*80}")
        
        if not results['ids']:
            print("Protocol not found.")
            return
        
        # Get protocol info from first chunk
        first_metadata = results['metadatas'][0]
        print(f"Source File: {first_metadata.get('source_file', 'N/A')}")
        print(f"Title: {first_metadata.get('title', 'N/A')}")
        print(f"Total Chunks: {first_metadata.get('total_chunks', 'N/A')}")
        print(f"ICD Codes: {first_metadata.get('icd_codes_str', 'N/A')}")
        
        print(f"\n{'-'*60}")
        print("PROTOCOL CONTENT")
        print(f"{'-'*60}")
        
        for i, (doc_id, document, metadata) in enumerate(zip(
            results['ids'],
            results['documents'],
            results['metadatas']
        )):
            print(f"\n--- CHUNK {metadata.get('chunk_number', i+1)} ---\n")
            print(document)


def main():
    """Main function to handle command line arguments and run the search."""
    parser = argparse.ArgumentParser(
        description="Search medical protocols in ChromaDB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search for protocols
  python protocol_search.py "HELLP syndrome treatment"
  
  # Search with specific protocol ID filter
  python protocol_search.py "diagnosis" --protocol-id p_d57148b2d4
  
  # Get all chunks for a specific protocol
  python protocol_search.py --get-protocol p_d57148b2d4
  
  # List all protocols
  python protocol_search.py --list-protocols
  
  # Search with custom database settings
  python protocol_search.py "pregnancy" --db-dir ./my_db --collection-name my_protocols
        """
    )
    
    parser.add_argument(
        "query",
        nargs='?',
        help="Search query text (not required for --get-protocol or --list-protocols)"
    )
    
    parser.add_argument(
        "--n-results",
        type=int,
        default=5,
        help="Number of results to return (default: 5)"
    )
    
    parser.add_argument(
        "--protocol-id",
        help="Filter results by specific protocol ID"
    )
    
    parser.add_argument(
        "--get-protocol",
        help="Get all chunks for a specific protocol ID"
    )
    
    parser.add_argument(
        "--list-protocols",
        action="store_true",
        help="List all protocol IDs in the collection"
    )
    
    parser.add_argument(
        "--db-dir",
        default="./chroma_db",
        help="Directory where ChromaDB is persisted (default: ./chroma_db)"
    )
    
    parser.add_argument(
        "--collection-name",
        default="medical_protocols",
        help="Name of the ChromaDB collection (default: medical_protocols)"
    )
    
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Name of the embedding model (default: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)"
    )
    
    args = parser.parse_args()
    
    # Create searcher
    searcher = ProtocolSearcher(
        chroma_persist_directory=args.db_dir,
        collection_name=args.collection_name,
        embedding_model_name=args.embedding_model
    )
    
    # Handle different operations
    if args.list_protocols:
        protocols = searcher.list_protocols()
        print(f"\nFound {len(protocols)} protocols:")
        for i, protocol_id in enumerate(protocols, 1):
            print(f"{i:3d}. {protocol_id}")
    
    elif args.get_protocol:
        results = searcher.get_protocol_chunks(args.get_protocol)
        searcher.print_protocol_chunks(args.get_protocol, results)
    
    elif args.query:
        # Prepare where clause if protocol ID is specified
        where_clause = {'protocol_id': args.protocol_id} if args.protocol_id else None
        
        # Perform search
        results = searcher.search(
            query=args.query,
            n_results=args.n_results,
            where=where_clause
        )
        
        # Print results
        searcher.print_search_results(results, args.query)
    
    else:
        parser.print_help()
        print("\nError: Please provide a search query or use --list-protocols or --get-protocol")


if __name__ == "__main__":
    main()