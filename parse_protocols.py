#!/usr/bin/env python3
"""
Fixed script for processing medical protocols and loading them into a vector database.
Splits protocols into chunks, generates embeddings, and stores them in ChromaDB.
This version properly handles RecursiveCharacterTextSplitter from LangChain.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import argparse

# Add necessary imports
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProtocolEmbeddingFunction(embedding_functions.EmbeddingFunction):
    """Custom embedding function for protocols using transformers."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", chunk_size: int = 512):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Try to load on GPU first, fallback to CPU if out of memory
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_tokens = chunk_size
        
        try:
            self.model = AutoModel.from_pretrained(model_name)
            if self.device == "cuda":
                try:
                    self.model.to(self.device)
                    logger.info(f"Loaded embedding model {model_name} on {self.device}")
                except torch.cuda.OutOfMemoryError:
                    logger.warning(f"GPU out of memory, falling back to CPU for {model_name}")
                    self.device = "cpu"
                    self.model.to(self.device)
                    logger.info(f"Loaded embedding model {model_name} on {self.device}")
            else:
                self.model.to(self.device)
                logger.info(f"Loaded embedding model {model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        embeddings = []
        
        # Process in batches to avoid memory issues
        batch_size = 8
        for i in range(0, len(input), batch_size):
            batch_texts = input[i:i + batch_size]
            
            # Tokenize and generate embeddings
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=False,
                max_length=self.max_tokens,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                # Use mean pooling
                token_embeddings = model_output.last_hidden_state
                attention_mask = encoded_input['attention_mask']
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                batch_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                
            # Convert to numpy and normalize
            batch_embeddings = batch_embeddings.cpu().numpy()
            batch_embeddings = batch_embeddings / np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
            
            embeddings.extend(batch_embeddings.tolist())
        
        return embeddings


class ProtocolProcessor:
    """Class for processing medical protocols and storing them in ChromaDB."""
    
    def __init__(
        self,
        chroma_persist_directory: str = "./chroma_db",
        collection_name: str = "ChunkLength-512",
        embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        chunk_size: int = 512,
        chunk_overlap: int = 200
    ):
        """
        Initialize protocol processor.
        
        Args:
            chroma_persist_directory: Directory to persist ChromaDB
            collection_name: Name of ChromaDB collection
            embedding_model_name: Name of embedding model
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
        """
        self.chroma_persist_directory = chroma_persist_directory
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize embedding function
        self.embedding_function = ProtocolEmbeddingFunction(embedding_model_name, chunk_size)
        
        # Initialize text splitter
        self.text_splitter = self._create_text_splitter()
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=chroma_persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Initialized ProtocolProcessor with collection '{collection_name}'")
    
    def _create_text_splitter(self):
        """Create text splitter with proper token counting."""
        try:            
            def token_length_function(text: str) -> int:
                # Try different parameter names for different tokenizer versions
                try:
                    return len(self.embedding_function.tokenizer.encode(text, add_special_tokens=False))
                except TypeError:
                    try:
                        return len(self.embedding_function.tokenizer.encode(text))
                    except:
                        # Fallback: rough estimate (4 chars per token)
                        return len(text) // 4
            
            # Create the splitter with proper length function
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=token_length_function,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            logger.info("Using LangChain RecursiveCharacterTextSplitter")
            return text_splitter
            
        except ImportError:
            logger.warning("LangChain not available, using fallback splitter")
            return self._create_fallback_splitter()
        except Exception as e:
            logger.warning(f"LangChain splitter failed, using fallback: {e}")
            return self._create_fallback_splitter()
    
    def _create_fallback_splitter(self):
        """Create fallback text splitter when LangChain is not available."""
        class FallbackSplitter:
            def __init__(self, chunk_size, chunk_overlap):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap
                self.separators = ["\n\n", "\n", ". ", " ", ""]
            
            def split_text(self, text: str) -> List[str]:
                if not text:
                    return []
                
                chunks = []
                current_pos = 0
                
                while current_pos < len(text):
                    # Calculate end position for current chunk
                    end_pos = min(current_pos + self.chunk_size * 4, len(text))  # Rough estimate
                    
                    # Try to find a good breaking point
                    best_break = end_pos
                    for separator in self.separators:
                        sep_pos = text.rfind(separator, current_pos, end_pos)
                        if sep_pos > current_pos:
                            best_break = sep_pos + len(separator)
                            break
                    
                    # Extract chunk
                    chunk = text[current_pos:best_break]
                    
                    # Check token count
                    token_count = len(self.embedding_function.tokenizer.encode(chunk))
                    
                    if token_count <= self.chunk_size:
                        chunks.append(chunk)
                        # Move to next position with overlap
                        if best_break >= len(text):
                            break  # End of text
                        current_pos = max(current_pos + 1, best_break - self.chunk_overlap)
                    else:
                        # Chunk is too large, split it more aggressively
                        sub_chunks = self._split_large_chunk_fallback(chunk)
                        chunks.extend(sub_chunks)
                        # Move to next position with overlap
                        if best_break >= len(text):
                            break  # End of text
                        current_pos = max(current_pos + 1, best_break - self.chunk_overlap)
                
                return [chunk.strip() for chunk in chunks if chunk.strip()]
            
            def _split_large_chunk_fallback(self, text: str) -> List[str]:
                """Split a chunk that's too large."""
                chunks = []
                current_pos = 0
                
                while current_pos < len(text):
                    # Find position that gives us the right token count
                    end_pos = current_pos + self.chunk_size * 4  # Rough estimate
                    
                    while end_pos > current_pos:
                        chunk = text[current_pos:end_pos]
                        token_count = len(self.embedding_function.tokenizer.encode(chunk))
                        
                        if token_count <= self.chunk_size:
                            break
                        
                        # Reduce chunk size
                        end_pos = int(end_pos * 0.9)
                    
                    if end_pos <= current_pos:
                        # Force split if we can't find a good position
                        end_pos = current_pos + 100
                    
                    chunk = text[current_pos:end_pos]
                    if chunk.strip():
                        chunks.append(chunk.strip())
                    
                    current_pos = end_pos - self.chunk_overlap if end_pos < len(text) else end_pos
                
                return chunks
        
        return FallbackSplitter(self.chunk_size, self.chunk_overlap)
    
    def load_protocols_from_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load protocols from a JSONL file.
        
        Args:
            file_path: Path to JSONL file
            
        Returns:
            List of protocol dictionaries
        """
        protocols = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        protocol = json.loads(line)
                        protocols.append(protocol)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON on line {line_num}: {e}")
                        
        except FileNotFoundError:
            logger.error(f"File '{file_path}' not found.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            sys.exit(1)
        
        logger.info(f"Loaded {len(protocols)} protocols from {file_path}")
        return protocols
    
    def split_protocol_into_chunks(self, protocol: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a protocol into chunks.
        
        Args:
            protocol: Protocol dictionary
            
        Returns:
            List of chunk dictionaries
        """
        protocol_id = protocol.get('protocol_id', 'unknown')
        source_file = protocol.get('source_file', 'unknown')
        title = protocol.get('title', 'unknown')
        icd_codes = protocol.get('icd_codes', [])
        text = protocol.get('text', '')
        
        if not text:
            logger.warning(f"Protocol {protocol_id} has no text content")
            return []
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create chunk dictionaries
        chunk_dicts = []
        for i, chunk_text in enumerate(chunks):
            chunk_dict = {
                'id': f"{protocol_id}_chunk_{i}",
                'text': chunk_text,
                'metadata': {
                    'protocol_id': protocol_id,
                    'source_file': source_file,
                    'title': title,
                    'chunk_number': i,
                    'total_chunks': len(chunks),
                    'icd_codes_str': ','.join(icd_codes) if icd_codes else ''
                }
            }
            chunk_dicts.append(chunk_dict)
        
        logger.info(f"Split protocol {protocol_id} into {len(chunks)} chunks")
        return chunk_dicts
    
    def store_chunks_in_chroma(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Store document chunks in ChromaDB.
        
        Args:
            chunks: List of chunk dictionaries
        """
        if not chunks:
            logger.warning("No chunks to store")
            return
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            ids.append(chunk['id'])
            documents.append(chunk['text'])
            metadatas.append(chunk['metadata'])
        
        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_documents = documents[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            
            self.collection.add(
                ids=batch_ids,
                documents=batch_documents,
                metadatas=batch_metadatas
            )
            
            logger.info(f"Stored batch {i//batch_size + 1}: {len(batch_ids)} chunks")
        
        logger.info(f"Stored total of {len(chunks)} chunks in ChromaDB")
    
    def process_protocols_file(self, file_path: str) -> None:
        """
        Process a JSONL file containing protocols and store them in ChromaDB.
        
        Args:
            file_path: Path to JSONL file
        """
        logger.info(f"Processing protocols file: {file_path}")
        
        # Load protocols
        protocols = self.load_protocols_from_jsonl(file_path)
        
        if not protocols:
            logger.warning("No protocols found in file")
            return
        
        # Process each protocol
        all_chunks = []
        for i_proto, protocol in enumerate(protocols):
            chunks = self.split_protocol_into_chunks(protocol)
            all_chunks.extend(chunks)
            if i_proto % 10 == 0:
                print(f"Processed {i_proto}th from {len(protocols)} protocols")
            # if i_proto > 10:
            #     break
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        
        # Store chunks in ChromaDB
        if all_chunks:
            self.store_chunks_in_chroma(all_chunks)
            logger.info(f"Successfully processed and stored {len(protocols)} protocols")
        else:
            logger.warning("No chunks were created from protocols")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about ChromaDB collection."""
        count = self.collection.count()
        logger.info(f"Collection '{self.collection_name}' contains {count} documents")
        
        # Get a sample of documents to understand structure
        if count > 0:
            sample = self.collection.get(limit=5)
            logger.info(f"Sample document IDs: {sample['ids']}")
            logger.info(f"Sample metadata keys: {list(sample['metadatas'][0].keys()) if sample['metadatas'] else 'None'}")
        
        return {
            'collection_name': self.collection_name,
            'document_count': count,
            'persist_directory': self.chroma_persist_directory
        }


def main():
    """Main function to handle command line arguments and run processing."""
    parser = argparse.ArgumentParser(
        description="Process medical protocols and store them in ChromaDB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python protocol_processor_fixed.py TaskQazCode/protocols_corpus.jsonl
  python protocol_processor_fixed.py TaskQazCode/protocols_corpus.jsonl --chunk-size 4000
        """
    )
    
    parser.add_argument(
        "file_path",
        help="Path to JSONL file containing protocols"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Target chunk size in tokens (default: 512)"
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between chunks in tokens (default: 200)"
    )
    
    parser.add_argument(
        "--db-dir",
        default="./chroma_db",
        help="Directory to persist ChromaDB (default: ./chroma_db)"
    )
    
    parser.add_argument(
        "--collection-name",
        default="ChunkLength-512",
        help="Name of ChromaDB collection (default: ChunkLength-512)"
    )
    
    parser.add_argument(
        "--embedding-model",
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Name of embedding model (default: Qwen/Qwen3-Embedding-0.6B)"
    )
    
    args = parser.parse_args()
    
    # Create processor
    processor = ProtocolProcessor(
        chroma_persist_directory=args.db_dir,
        collection_name=args.collection_name,
        embedding_model_name=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    # Process file
    processor.process_protocols_file(args.file_path)
    
    # Show statistics
    stats = processor.get_collection_stats()
    print("\n" + "="*50)
    print("PROCESSING COMPLETE")
    print("="*50)
    print(f"Collection: {stats['collection_name']}")
    print(f"Documents stored: {stats['document_count']}")
    print(f"Database directory: {stats['persist_directory']}")


if __name__ == "__main__":
    main()