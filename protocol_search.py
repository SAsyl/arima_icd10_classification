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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
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
    
    def __init__(self, model_name: str = "Qwen/Qwen3-Reranker-0.6B", max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        
        # Load as CausalLM (Generative) instead of SequenceClassification
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto" # Handles the GPU/CPU/MPS logic automatically
        ).eval()

        # Pre-cache the token IDs for "yes" and "no"
        self.token_yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_no_id = self.tokenizer.convert_tokens_to_ids("no")

        # Template components
        self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.instruction = "Given a protocol query, retrieve relevant technical sections that answer the query."

    def _format_input(self, query: str, doc: str):
        content = f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}"
        full_prompt = f"{self.prefix}{content}{self.suffix}"
        return full_prompt

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
        
        scores = []
        with torch.no_grad():
            for doc in docs:
                prompt = self._format_input(query_text, doc)
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                
                # Get logits for the very last token generated
                logits = self.model(**inputs).logits[:, -1, :]
                
                # Isolate "yes" vs "no"
                relevant_logits = torch.stack([logits[:, self.token_no_id], logits[:, self.token_yes_id]], dim=1)
                probs = torch.softmax(relevant_logits, dim=1)
                
                # The score is the probability of "yes" (index 1)
                score = probs[0, 1].item()
                scores.append(score)

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
        collection_name: str = "ChunkLength-512",
        embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        sqlite_db_path: str = "protocols.db",
        use_reranker: bool = True,
        reranker_model_name: str = "Qwen/Qwen3-Reranker-0.6B",
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
        self.use_reranker = use_reranker
        self.reranker_model_name = reranker_model_name
        
        # Initialize embedding function
        self.embedding_function = ProtocolEmbeddingFunction(embedding_model_name)
        self.reranker = None
        if self.use_reranker:
            try:
                self.reranker = ProtocolReranker(model_name=self.reranker_model_name)
            except Exception as e:
                logger.warning(
                    "Failed to initialize reranker '%s': %s. Using base similarity ranking.",
                    self.reranker_model_name,
                    e,
                )
                self.reranker = None
        
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

    def advanced_search(self, query: str, instruction: str = "Given a medical query, find relevant protocol chunks: ", top_k_chunks: int = 50, final_top_protocols: int = 3) -> List[Dict[str, Any]]:
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

        # Step 2: Deep reranking with Cross-Encoder (or fallback to base similarity)
        if self.reranker is not None:
            reranked_chunks = self.reranker.rerank(
                query_text=query, 
                initial_results=initial_results, 
                top_k=top_k_chunks # Rerank all of them to find the true best
            )
        else:
            reranked_chunks = []
            for doc_id, doc, metadata, distance in zip(
                initial_results.get("ids", [[]])[0],
                initial_results.get("documents", [[]])[0],
                initial_results.get("metadatas", [[]])[0],
                initial_results.get("distances", [[]])[0],
            ):
                reranked_chunks.append(
                    {
                        "id": doc_id,
                        "document": doc,
                        "metadata": metadata,
                        "rerank_score": 1.0 - float(distance),
                        "original_distance": distance,
                    }
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
    
    def describe_database_structure(self) -> Dict[str, Any]:
        """
        Describe the structure of the ChromaDB database, including all collections.
        
        Returns:
            Dictionary containing database structure information
        """
        logger.info("Describing ChromaDB database structure")
        
        try:
            # Get all collections in the database
            collections = self.client.list_collections()
            
            db_structure = {
                "database_path": self.chroma_persist_directory,
                "total_collections": len(collections),
                "collections": []
            }
            
            for collection in collections:
                collection_info = {
                    "name": collection.name,
                    "id": collection.id,
                    "metadata": collection.metadata
                }
                
                # Get collection statistics
                try:
                    # Get count of items in collection
                    collection_count = collection.count()
                    collection_info["item_count"] = collection_count
                    
                    # Get a sample of items to understand the structure
                    if collection_count > 0:
                        sample_items = collection.get(limit=5, include=['metadatas', 'documents'])
                        
                        # Analyze metadata structure
                        metadata_keys = set()
                        for metadata in sample_items['metadatas']:
                            if metadata:
                                metadata_keys.update(metadata.keys())
                        
                        collection_info["metadata_fields"] = list(metadata_keys)
                        
                        # Get document statistics
                        if sample_items['documents']:
                            doc_lengths = [len(doc) for doc in sample_items['documents'] if doc]
                            if doc_lengths:
                                collection_info["document_stats"] = {
                                    "sample_count": len(doc_lengths),
                                    "min_length": min(doc_lengths),
                                    "max_length": max(doc_lengths),
                                    "avg_length": sum(doc_lengths) / len(doc_lengths)
                                }
                        
                        # Check for embedding function
                        if hasattr(collection, '_embedding_function'):
                            collection_info["has_embedding_function"] = True
                        else:
                            collection_info["has_embedding_function"] = False
                            
                    else:
                        collection_info["item_count"] = 0
                        collection_info["metadata_fields"] = []
                        collection_info["has_embedding_function"] = False
                        
                except Exception as e:
                    logger.warning(f"Error getting details for collection {collection.name}: {e}")
                    collection_info["error"] = str(e)
                
                db_structure["collections"].append(collection_info)
            
            logger.info(f"Database contains {len(collections)} collections")
            return db_structure
            
        except Exception as e:
            logger.error(f"Failed to describe database structure: {e}")
            return {
                "database_path": self.chroma_persist_directory,
                "error": str(e),
                "collections": []
            }
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection from the ChromaDB database.
        
        Args:
            collection_name: Name of the collection to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        logger.info(f"Attempting to delete collection: {collection_name}")
        
        try:
            # Check if collection exists
            collections = self.client.list_collections()
            collection_exists = any(c.name == collection_name for c in collections)
            
            if not collection_exists:
                logger.warning(f"Collection '{collection_name}' does not exist")
                return False
            
            # Delete the collection
            self.client.delete_collection(name=collection_name)
            logger.info(f"Successfully deleted collection: {collection_name}")
            
            # If we deleted the current collection, set it to None
            if self.collection_name == collection_name:
                self.collection = None
                logger.warning(f"Current collection '{collection_name}' was deleted. Please reconnect to a different collection.")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete collection '{collection_name}': {e}")
            return False
    
    def print_database_structure(self, db_structure: Dict[str, Any]) -> None:
        """
        Print the database structure in a readable format.
        
        Args:
            db_structure: Database structure information from describe_database_structure
        """
        print(f"\n{'='*80}")
        print("CHROMADB DATABASE STRUCTURE")
        print(f"{'='*80}")
        print(f"Database Path: {db_structure['database_path']}")
        print(f"Total Collections: {db_structure['total_collections']}")
        
        if 'error' in db_structure:
            print(f"Error: {db_structure['error']}")
            return
        
        for i, collection in enumerate(db_structure['collections'], 1):
            print(f"\n{'-'*60}")
            print(f"COLLECTION {i}: {collection['name']}")
            print(f"{'-'*60}")
            print(f"ID: {collection['id']}")
            print(f"Item Count: {collection.get('item_count', 'N/A')}")
            print(f"Has Embedding Function: {collection.get('has_embedding_function', 'N/A')}")
            
            if collection.get('metadata'):
                print(f"Collection Metadata: {collection['metadata']}")
            
            if collection.get('metadata_fields'):
                print(f"Metadata Fields: {', '.join(collection['metadata_fields'])}")
            
            if collection.get('document_stats'):
                stats = collection['document_stats']
                print(f"Document Statistics (sample of {stats['sample_count']} items):")
                print(f"  - Min Length: {stats['min_length']} characters")
                print(f"  - Max Length: {stats['max_length']} characters")
                print(f"  - Avg Length: {stats['avg_length']:.1f} characters")
            
            if 'error' in collection:
                print(f"Error: {collection['error']}")
    
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


    def print_advanced_search_results(self, query: str, results: List[Dict[str, Any]]) -> None:
        """
        Print the fully ranked protocol results in a readable terminal format.
        
        Args:
            query: The original search query
            results: The list of dictionaries returned by advanced_search()
        """
        print(f"\n{'='*80}")
        print(f"ADVANCED SEARCH RESULTS FOR: '{query}'")
        print(f"{'='*80}")
        
        if not results:
            print("No protocols found.")
            return
        
        for i, result in enumerate(results):
            protocol_id = result.get('protocol_id', 'Unknown ID')
            score = result.get('rerank_score', 0.0)
            snippet = result.get('winning_chunk_snippet', '')
            
            # Extract the full document data fetched from SQLite
            full_data = result.get('full_protocol_data', {})
            source_file = full_data.get('source_file', 'N/A')
            title = full_data.get('title', 'N/A')
            
            # ICD codes might be a list or a string depending on your JSON structure
            icd_codes = full_data.get('icd_codes', 'N/A')
            if isinstance(icd_codes, list):
                icd_codes = ", ".join(icd_codes) if icd_codes else "None"
                
            full_text = full_data.get('text', '')
            
            print(f"\nRANK {i+1} | Score: {score:.4f} | Protocol: {protocol_id}")
            print(f"{'-'*80}")
            print(f"Source File : {source_file}")
            print(f"Title       : {title}")
            print(f"ICD Codes   : {icd_codes}")
            
            print(f"\n>> BEST MATCHING SNIPPET (Why this protocol was chosen):")
            print(f"{snippet}")
            
            print(f"\n>> FULL PROTOCOL TEXT PREVIEW (First 300 chars...):")
            # Print just the beginning of the full text so it doesn't flood your terminal
            preview_text = full_text[:300].replace('\n', ' ')
            print(f"{preview_text}...")
            print(f"{'-'*80}")

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
  
  # Describe database structure
  python protocol_search.py --describe-db
  
  # Delete a collection
  python protocol_search.py --delete-collection MyCollection
  
  # Search with custom database settings
  python protocol_search.py "pregnancy" --db-dir ./my_db --collection-name my_protocols
        """
    )
    
    parser.add_argument(
        "query",
        nargs='?',
        help="Search query text (not required for --get-protocol, --list-protocols, --describe-db, or --delete-collection)"
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
        "--describe-db",
        action="store_true",
        help="Describe the structure of the ChromaDB database"
    )
    
    parser.add_argument(
        "--delete-collection",
        help="Delete a collection from the database"
    )
    
    parser.add_argument(
        "--db-dir",
        default="./chroma_db",
        help="Directory where ChromaDB is persisted (default: ./chroma_db)"
    )
    
    parser.add_argument(
        "--collection-name",
        default="ChunkLength-512",
        help="Name of the ChromaDB collection (default: ChunkLength-512)"
    )
    
    parser.add_argument(
        "--embedding-model",
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Name of the embedding model (default: Qwen/Qwen3-Embedding-0.6B)"
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
    if args.describe_db:
        db_structure = searcher.describe_database_structure()
        searcher.print_database_structure(db_structure)
    
    elif args.delete_collection:
        success = searcher.delete_collection(args.delete_collection)
        if success:
            print(f"Successfully deleted collection: {args.delete_collection}")
        else:
            print(f"Failed to delete collection: {args.delete_collection}")
    
    elif args.list_protocols:
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

        # # This will return a clean list of the top 3 FULL protocols, ranked accurately!
        best_protocols = searcher.advanced_search(args.query)

        searcher.print_advanced_search_results(args.query, best_protocols)
    
    else:
        parser.print_help()
        print("\nError: Please provide a search query or use --list-protocols, --get-protocol, --describe-db, or --delete-collection")


if __name__ == "__main__":
    main()
