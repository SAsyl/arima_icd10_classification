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
from parse_protocols import ProtocolEmbeddingFunction
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sqlite3
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProtocolReranker:
    """Custom reranking function for protocols using a cross-encoder."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-Reranker-0.6B", max_length: int = 2048):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Try to load on GPU first, check for Apple Silicon, fallback to CPU
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        try:
            # Load Model in float16 for 4GB VRAM safety.
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=1,
                torch_dtype=torch.float16
            )
            
            if self.device == "cuda":
                try:
                    self.model.to(self.device)
                    logger.info(f"Loaded reranker model {model_name} on {self.device}")
                except torch.cuda.OutOfMemoryError:
                    logger.warning(f"GPU out of memory, falling back to CPU for {model_name}")
                    self.device = "cpu"
                    self.model.to(self.device)
                    self.model.float() # Float32 is safer/faster for CPU inference
                    logger.info(f"Loaded reranker model {model_name} on {self.device}")
            else:
                self.model.to(self.device)
                if self.device == "cpu":
                    self.model.float() 
                logger.info(f"Loaded reranker model {model_name} on {self.device}")
                
            # Set model to evaluation mode
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def rerank(self, query_text: str, initial_results: Dict[str, Any], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Takes the initial results from ChromaDB and reranks them using a Cross-Encoder.
        Processes one pair at a time to prevent VRAM overflow.
        """
        if not initial_results or not initial_results.get('ids') or not initial_results['ids'][0]:
            logger.warning("No results to rerank.")
            return []

        # Extract data from Chroma's dictionary output
        docs = initial_results['documents'][0]
        metadatas = initial_results['metadatas'][0]
        ids = initial_results['ids'][0]
        distances = initial_results['distances'][0]

        scores = []
        logger.info("Calculating Cross-Encoder scores...")
        
        # Process one Query-Document pair at a time (Batch Size = 1)
        with torch.no_grad():
            for doc in docs:
                # Tokenize pair
                inputs = self.tokenizer(
                    query_text,
                    doc,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length, 
                    return_tensors="pt"
                ).to(self.device)
                
                # Predict
                output = self.model(**inputs)
                
                # Extract the raw logit score
                score = float(output.logits.squeeze())
                scores.append(score)
                
                # Aggressive memory clearing for 4GB VRAM
                if self.device == "cuda":
                    torch.cuda.empty_cache()

        # Sort by Cross-Encoder score in descending order
        scores_array = np.array(scores)
        ranked_indices = np.argsort(scores_array)[::-1]

        reranked_output = []
        
        # Format the final top_k results
        for i, idx in enumerate(ranked_indices[:top_k]):
            doc_id = ids[idx]
            score = scores[idx]
            original_dist = distances[idx]
            metadata = metadatas[idx]
            document = docs[idx]
            
            logger.debug(f"Rank {i+1}: Chunk ID {doc_id} | Score: {score:.4f}")
            
            reranked_output.append({
                "id": doc_id,
                "document": document,
                "metadata": metadata,
                "rerank_score": score,
                "original_distance": original_dist
            })
            
        return reranked_output


class ProtocolSearcher:
    """Class for searching medical protocols in ChromaDB."""
    
    def __init__(
        self,
        chroma_persist_directory: str = "./chroma_db",
        collection_name: str = "medical_protocols",
        embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        sqlite_db_path: str = "protocols.db"
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
        self.sqlite_db_path = sqlite_db_path
        
        # Initialize embedding function
        self.embedding_function = ProtocolEmbeddingFunction(embedding_model_name)
        self.reranker = ProtocolReranker()
        
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
    
    def index_full_protocols_to_sqlite(self, jsonl_file: str):
        """
        Reads the original JSONL file and stores the full documents in SQLite 
        for instant retrieval by protocol_id.
        """
        logger.info(f"Indexing full protocols into SQLite DB: {self.sqlite_db_path}")
        conn = sqlite3.connect(self.sqlite_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS protocols (
                protocol_id TEXT PRIMARY KEY,
                source_file TEXT,
                title TEXT,
                data JSON
            )
        ''')
        
        count = 0
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                protocol_id = obj.get('protocol_id')
                
                if protocol_id:
                    cursor.execute('''
                        INSERT OR REPLACE INTO protocols (protocol_id, source_file, title, data)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        protocol_id, 
                        obj.get('source_file'), 
                        obj.get('title'), 
                        json.dumps(obj)
                    ))
                    count += 1
        
        conn.commit()
        conn.close()
        logger.info(f"Successfully indexed {count} protocols into SQLite.")

    def get_full_protocol(self, protocol_id: str) -> Optional[Dict[str, Any]]:
        """
        Instantly fetch the full original protocol JSON from SQLite.
        """
        conn = sqlite3.connect(self.sqlite_db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT data FROM protocols WHERE protocol_id = ?", (protocol_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return json.loads(row[0])
        else:
            logger.warning(f"Protocol ID '{protocol_id}' not found in SQLite.")
            return None

    def advanced_search(self, query: str, instruction: str = "Given a medical query, find relevant protocol chunks: ", top_k_chunks: int = 15, final_top_protocols: int = 3) -> List[Dict[str, Any]]:
        """
        The complete pipeline: 
        1. Chroma Search -> 2. Qwen Rerank -> 3. Max Pooling -> 4. SQLite Full Fetch
        """
        # Step 1: Broad search in ChromaDB
        initial_results = self.search(
            query=query, 
            n_results=top_k_chunks, 
            instruction=instruction
        )
        
        if not initial_results['ids'][0]:
            return []

        # Step 2: Deep reranking with Cross-Encoder
        reranked_chunks = self.reranker.rerank(
            query_text=query, 
            initial_results=initial_results, 
            top_k=top_k_chunks # Rerank all of them to find the true best
        )

        # Step 3: Max Pooling (Get the highest scoring chunk per protocol_id)
        protocol_dict = {}
        for chunk in reranked_chunks:
            pid = chunk['metadata'].get('protocol_id')
            score = chunk['rerank_score']
            
            if pid and (pid not in protocol_dict or score > protocol_dict[pid]['best_score']):
                protocol_dict[pid] = {
                    'protocol_id': pid,
                    'best_score': score,
                    'winning_chunk': chunk['document']
                }

        # Sort protocols by their best chunk's score
        ranked_pids = sorted(list(protocol_dict.values()), key=lambda x: x['best_score'], reverse=True)
        
        # Step 4: Fetch full data from SQLite for the top N protocols
        final_output = []
        for p in ranked_pids[:final_top_protocols]:
            full_data = self.get_full_protocol(p['protocol_id'])
            if full_data:
                final_output.append({
                    "protocol_id": p['protocol_id'],
                    "rerank_score": p['best_score'],
                    "winning_chunk_snippet": p['winning_chunk'][:200] + "...",
                    "full_protocol_data": full_data # The complete JSON from SQLite!
                })

        return final_output

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        instruction: Optional[str] = None
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
                query_texts=[instruction + query],
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
            print(document[:500] + "..." if len(document) > 500 else document)
            # print(len(document), document)
    
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

    parser.add_argument(
        "--query-instruction",
        default="Given a search query, retrieve relevant passages that answer the query: ",
        help="Instruction to paste before query"
    )
    
    args = parser.parse_args()
    
    # Create searcher
    searcher = ProtocolSearcher(
        chroma_persist_directory=args.db_dir,
        collection_name=args.collection_name,
        embedding_model_name=args.embedding_model,
        sqlite_db_path="protocols.db"
    )
    
    searcher.index_full_protocols_to_sqlite("TaskQazCode/protocols_corpus.jsonl")

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
        
        # # Perform search
        # results = searcher.search(
        #     query=args.query,
        #     n_results=args.n_results,
        #     where=where_clause,
        #     instruction=args.query_instruction
        # )
        

        # # Print results
        # searcher.print_search_results(results, args.query)

        # This will return a clean list of the top 3 FULL protocols, ranked accurately!
        best_protocols = searcher.advanced_search(args.query)

        print(best_protocols)
    
    else:
        parser.print_help()
        print("\nError: Please provide a search query or use --list-protocols or --get-protocol")


if __name__ == "__main__":
    main()