"""
Vector Database Manager v4
- Semantic chunking (keeps concepts together) OR fixed chunking
- Progress bars with ETA for all operations
- PDF support (via PyMuPDF)
- Batch processing with visible progress
- Memory usage display
- Time tracking for each phase
- Graceful cancellation (Ctrl+C)

New in v4:
- Semantic chunking option that uses embeddings to detect natural topic breaks
- Keeps full concepts (like spell descriptions, rules) intact
- Better for rulebooks, structured content, and reference material
"""

import os
import sys
import json
import time
import re
import threading
from pathlib import Path
from typing import List, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

# ============================================================================
# Configuration
# ============================================================================

LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
VECTOR_DB_DIR = Path("./vector_db")
COLLECTION_NAME = "long_term_memory"

# Batch sizes for progress display
EMBEDDING_BATCH_SIZE = 10      # Show progress every N embeddings
DB_INSERT_BATCH_SIZE = 50      # Insert to DB in batches of N

# Semantic chunking settings
SEMANTIC_SIMILARITY_THRESHOLD = 0.5  # Lower = more splits (0.3-0.7 typical)
MAX_CHUNK_SIZE = 1500                 # Max chars per chunk (even semantic)
MIN_CHUNK_SIZE = 100                  # Min chars per chunk

# ============================================================================
# Chunking Mode Enum
# ============================================================================

class ChunkingMode(Enum):
    FIXED = "fixed"           # Traditional fixed-size with overlap
    SEMANTIC = "semantic"     # Smart splitting by topic/meaning
    SENTENCE = "sentence"     # Split by sentences (middle ground)

# ============================================================================
# Progress Bar Utilities
# ============================================================================

def format_time(seconds: float) -> str:
    """Format seconds into human readable time"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = seconds // 60
        secs = seconds % 60
        return f"{mins:.0f}m {secs:.0f}s"
    else:
        hours = seconds // 3600
        mins = (seconds % 3600) // 60
        return f"{hours:.0f}h {mins:.0f}m"

def format_bytes(bytes_val: int) -> str:
    """Format bytes into human readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024:
            return f"{bytes_val:.1f}{unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f}TB"

class ProgressBar:
    """
    A rich progress bar with ETA, speed, and customizable display.
    Works in terminal without external dependencies.
    """
    
    def __init__(self, total: int, desc: str = "", width: int = 40, 
                 show_eta: bool = True, show_speed: bool = True):
        self.total = total
        self.current = 0
        self.desc = desc
        self.width = width
        self.show_eta = show_eta
        self.show_speed = show_speed
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.last_update_count = 0
        self._lock = threading.Lock()
        self._cancelled = False
        
    def update(self, n: int = 1, suffix: str = ""):
        """Update progress by n items"""
        with self._lock:
            self.current += n
            self._render(suffix)
    
    def set(self, value: int, suffix: str = ""):
        """Set progress to specific value"""
        with self._lock:
            self.current = value
            self._render(suffix)
    
    def _render(self, suffix: str = ""):
        """Render the progress bar"""
        now = time.time()
        elapsed = now - self.start_time
        
        # Calculate percentage and bar
        if self.total > 0:
            percent = min(100, (self.current / self.total) * 100)
            filled = int(self.width * self.current / self.total)
        else:
            percent = 0
            filled = 0
        
        bar = "█" * filled + "░" * (self.width - filled)
        
        # Calculate speed (items per second)
        speed = self.current / elapsed if elapsed > 0 else 0
        
        # Calculate ETA
        if speed > 0 and self.current < self.total:
            remaining = (self.total - self.current) / speed
            eta_str = format_time(remaining)
        else:
            eta_str = "--"
        
        # Build the status line
        status = f"\r{self.desc}: |{bar}| {self.current}/{self.total} ({percent:.1f}%)"
        
        if self.show_speed:
            status += f" [{speed:.1f}/s]"
        
        if self.show_eta and self.current < self.total:
            status += f" ETA: {eta_str}"
        elif self.current >= self.total:
            status += f" Done in {format_time(elapsed)}"
        
        if suffix:
            status += f" | {suffix}"
        
        # Clear to end of line and print
        status += " " * 10  # Padding to clear previous content
        print(status, end="", flush=True)
    
    def finish(self, message: str = ""):
        """Complete the progress bar"""
        self.current = self.total
        self._render()
        elapsed = time.time() - self.start_time
        if message:
            print(f"\n✓ {message} ({format_time(elapsed)})")
        else:
            print()  # New line
    
    def cancel(self):
        """Mark as cancelled"""
        self._cancelled = True
        print(f"\n⚠ Cancelled at {self.current}/{self.total}")

class SpinnerProgress:
    """
    A spinner for operations where total is unknown.
    """
    
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    
    def __init__(self, desc: str = "Processing"):
        self.desc = desc
        self.count = 0
        self.start_time = time.time()
        self.frame_idx = 0
        self._running = False
        self._thread = None
    
    def start(self):
        """Start the spinner"""
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
    
    def _spin(self):
        """Spin animation thread"""
        while self._running:
            elapsed = time.time() - self.start_time
            frame = self.FRAMES[self.frame_idx % len(self.FRAMES)]
            status = f"\r{frame} {self.desc}... ({format_time(elapsed)}) [{self.count} items]"
            print(status + " " * 10, end="", flush=True)
            self.frame_idx += 1
            time.sleep(0.1)
    
    def update(self, n: int = 1):
        """Update item count"""
        self.count += n
    
    def stop(self, message: str = ""):
        """Stop the spinner"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.5)
        elapsed = time.time() - self.start_time
        if message:
            print(f"\r✓ {message} ({self.count} items, {format_time(elapsed)})" + " " * 20)
        else:
            print("\r" + " " * 60 + "\r", end="")

# ============================================================================
# Dependency Checks
# ============================================================================

def check_dependencies():
    """Check and report on dependencies"""
    missing = []
    optional_missing = []
    
    # Required
    try:
        from langchain_chroma import Chroma
    except ImportError:
        missing.append("langchain-chroma chromadb")
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        missing.append("langchain-huggingface sentence-transformers")
    
    # Optional but recommended
    try:
        import fitz  # PyMuPDF
    except ImportError:
        optional_missing.append("pymupdf (for PDF support)")
    
    try:
        import psutil
    except ImportError:
        optional_missing.append("psutil (for memory monitoring)")
    
    try:
        import numpy as np
    except ImportError:
        optional_missing.append("numpy (for semantic chunking)")
    
    if missing:
        print("\n❌ Missing required packages:")
        for pkg in missing:
            print(f"   pip install {pkg}")
        sys.exit(1)
    
    if optional_missing:
        print("\n⚠ Optional packages (recommended):")
        for pkg in optional_missing:
            print(f"   pip install {pkg}")
        print()

check_dependencies()

# ============================================================================
# Imports (after dependency check)
# ============================================================================

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

# Optional imports
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# ============================================================================
# Memory Monitoring
# ============================================================================

def get_memory_usage() -> str:
    """Get current memory usage"""
    if PSUTIL_AVAILABLE:
        process = psutil.Process()
        mem = process.memory_info().rss
        return format_bytes(mem)
    return "N/A"

def print_system_stats():
    """Print current system resource usage"""
    if PSUTIL_AVAILABLE:
        process = psutil.Process()
        mem = process.memory_info().rss
        cpu = process.cpu_percent(interval=0.1)
        print(f"   💾 Memory: {format_bytes(mem)} | CPU: {cpu:.1f}%")

# ============================================================================
# Text Utilities for Chunking
# ============================================================================

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences. Handles common abbreviations and edge cases.
    """
    # Handle common abbreviations that shouldn't split
    text = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|Jr|Sr|vs|etc|i\.e|e\.g)\.\s', r'\1<DOT> ', text)
    
    # Split on sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Restore dots
    sentences = [s.replace('<DOT>', '.') for s in sentences]
    
    # Filter empty and clean up
    return [s.strip() for s in sentences if s.strip()]

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    if NUMPY_AVAILABLE:
        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
    else:
        # Pure Python fallback
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        return dot / (norm1 * norm2 + 1e-10)

# ============================================================================
# LM Studio Embedding Class (with progress)
# ============================================================================

class LMStudioEmbeddings(Embeddings):
    """
    Custom embedding class that uses LM Studio's API.
    Requires an embedding model loaded in LM Studio.
    """
    
    def __init__(self, base_url: str = LM_STUDIO_BASE_URL, timeout: int = 60):
        self.base_url = base_url
        self.endpoint = f"{base_url}/embeddings"
        self.timeout = timeout
        self._dimension = None
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        import requests
        
        try:
            response = requests.post(
                self.endpoint,
                headers={"Content-Type": "application/json"},
                json={"input": text, "model": "embedding-model"},
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                embedding = data['data'][0]['embedding']
                self._dimension = len(embedding)
                return embedding
            else:
                print(f"\n⚠ LM Studio error: {response.status_code}")
                return [0.0] * (self._dimension or 384)
                
        except requests.exceptions.ConnectionError:
            print("\n⚠ Cannot connect to LM Studio. Is it running?")
            return [0.0] * (self._dimension or 384)
        except requests.exceptions.Timeout:
            print("\n⚠ LM Studio timeout - try increasing timeout or reducing batch size")
            return [0.0] * (self._dimension or 384)
        except Exception as e:
            print(f"\n⚠ Embedding error: {e}")
            return [0.0] * (self._dimension or 384)
    
    def embed_documents(self, texts: List[str], show_progress: bool = True) -> List[List[float]]:
        """Embed a list of documents with optional progress bar"""
        embeddings = []
        
        if show_progress and len(texts) > 5:
            progress = ProgressBar(len(texts), desc="🔢 Embedding", show_speed=True)
            try:
                for i, text in enumerate(texts):
                    embedding = self._get_embedding(text)
                    embeddings.append(embedding)
                    progress.update(1, suffix=f"Mem: {get_memory_usage()}")
                progress.finish("Embeddings complete")
            except KeyboardInterrupt:
                progress.cancel()
                raise
        else:
            for text in texts:
                embeddings.append(self._get_embedding(text))
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query string"""
        return self._get_embedding(text)

# ============================================================================
# HuggingFace Embeddings Wrapper (with progress)
# ============================================================================

class ProgressHuggingFaceEmbeddings:
    """
    Wrapper around HuggingFaceEmbeddings that shows progress.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        print(f"📥 Loading embedding model: {model_name}")
        print("   (This may take a moment on first run...)")
        
        load_start = time.time()
        self._embeddings = HuggingFaceEmbeddings(model_name=model_name)
        load_time = time.time() - load_start
        
        print(f"   ✓ Model loaded in {format_time(load_time)}")
        print_system_stats()
    
    def embed_documents(self, texts: List[str], show_progress: bool = True) -> List[List[float]]:
        """Embed documents with progress tracking"""
        total = len(texts)
        
        if total <= 10 or not show_progress:
            # Small batch - just do it directly
            if show_progress:
                print(f"🔢 Embedding {total} texts...")
            return self._embeddings.embed_documents(texts)
        
        # For larger batches, process in chunks with progress
        batch_size = 32  # HuggingFace works well with batches of 32
        all_embeddings = []
        
        progress = ProgressBar(total, desc="🔢 Embedding", show_speed=True)
        
        try:
            for i in range(0, total, batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self._embeddings.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
                progress.update(len(batch), suffix=f"Mem: {get_memory_usage()}")
            
            progress.finish("Embeddings complete")
        except KeyboardInterrupt:
            progress.cancel()
            raise
        
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query"""
        return self._embeddings.embed_query(text)

# ============================================================================
# Chunking Functions
# ============================================================================

def chunk_text_fixed(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    FIXED CHUNKING: Split text into overlapping chunks of fixed size.
    Fast but may split concepts mid-sentence.
    
    Best for: Large documents where speed matters, general text
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence/paragraph boundary
        if end < len(text):
            for sep in ['\n\n', '\n', '. ', '! ', '? ']:
                idx = chunk.rfind(sep)
                if idx > chunk_size * 0.5:
                    chunk = chunk[:idx + len(sep)]
                    end = start + len(chunk)
                    break
        
        if chunk.strip():
            chunks.append(chunk.strip())
        
        start = end - overlap
    
    return chunks


def chunk_text_sentences(text: str, sentences_per_chunk: int = 5, overlap_sentences: int = 1) -> List[str]:
    """
    SENTENCE CHUNKING: Group sentences together.
    Middle ground between fixed and semantic.
    
    Best for: Narrative text, articles, general content
    """
    sentences = split_into_sentences(text)
    
    if not sentences:
        return [text] if text.strip() else []
    
    chunks = []
    i = 0
    
    while i < len(sentences):
        # Get sentences for this chunk
        end = min(i + sentences_per_chunk, len(sentences))
        chunk_sentences = sentences[i:end]
        chunk = ' '.join(chunk_sentences)
        
        if chunk.strip():
            chunks.append(chunk.strip())
        
        # Move forward, accounting for overlap
        i = end - overlap_sentences
        if i <= chunks[-1] if chunks else 0:  # Prevent infinite loop
            i = end
    
    return chunks


def chunk_text_semantic(text: str, embeddings, 
                        similarity_threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
                        max_chunk_size: int = MAX_CHUNK_SIZE,
                        min_chunk_size: int = MIN_CHUNK_SIZE,
                        show_progress: bool = False) -> List[str]:
    """
    SEMANTIC CHUNKING: Split text based on meaning/topic changes.
    Uses embeddings to detect where topics shift.
    
    Best for: Rulebooks, reference material, structured content
    
    How it works:
    1. Split text into sentences
    2. Embed each sentence
    3. Compare adjacent sentences' similarity
    4. Split where similarity drops below threshold
    5. Merge very small chunks, split very large ones
    """
    # Split into sentences
    sentences = split_into_sentences(text)
    
    if len(sentences) <= 2:
        # Too short for semantic analysis
        return [text] if text.strip() else []
    
    # Get embeddings for all sentences (quiet mode for chunking phase)
    if show_progress:
        print(f"   📊 Analyzing {len(sentences)} sentences for topic breaks...")
    
    try:
        # Use the embeddings model - call with show_progress=False to avoid nested progress bars
        if hasattr(embeddings, 'embed_documents'):
            sentence_embeddings = embeddings.embed_documents(sentences, show_progress=False)
        elif hasattr(embeddings, '_embeddings'):
            sentence_embeddings = embeddings._embeddings.embed_documents(sentences)
        else:
            sentence_embeddings = embeddings.embed_documents(sentences)
    except Exception as e:
        print(f"   ⚠ Semantic analysis failed ({e}), falling back to sentence chunking")
        return chunk_text_sentences(text)
    
    # Find breakpoints based on similarity drops
    breakpoints = []
    for i in range(len(sentences) - 1):
        similarity = cosine_similarity(sentence_embeddings[i], sentence_embeddings[i + 1])
        if similarity < similarity_threshold:
            breakpoints.append(i + 1)
    
    # Create chunks from breakpoints
    chunks = []
    start = 0
    
    for bp in breakpoints:
        chunk = ' '.join(sentences[start:bp])
        if chunk.strip():
            chunks.append(chunk.strip())
        start = bp
    
    # Don't forget the last chunk
    if start < len(sentences):
        chunk = ' '.join(sentences[start:])
        if chunk.strip():
            chunks.append(chunk.strip())
    
    # Post-process: merge tiny chunks, split huge ones
    final_chunks = []
    buffer = ""
    
    for chunk in chunks:
        if len(buffer) + len(chunk) < min_chunk_size:
            # Chunk too small, buffer it
            buffer = (buffer + " " + chunk).strip()
        elif len(buffer) > 0:
            # Flush buffer first
            if len(buffer) >= min_chunk_size:
                final_chunks.append(buffer)
            else:
                chunk = (buffer + " " + chunk).strip()
            buffer = ""
            
            # Handle current chunk
            if len(chunk) > max_chunk_size:
                # Split large chunk with fixed chunking
                final_chunks.extend(chunk_text_fixed(chunk, max_chunk_size, 50))
            else:
                final_chunks.append(chunk)
        else:
            if len(chunk) > max_chunk_size:
                final_chunks.extend(chunk_text_fixed(chunk, max_chunk_size, 50))
            else:
                final_chunks.append(chunk)
    
    # Flush any remaining buffer
    if buffer.strip():
        if final_chunks and len(buffer) < min_chunk_size:
            # Append to last chunk if buffer is tiny
            final_chunks[-1] = final_chunks[-1] + " " + buffer
        else:
            final_chunks.append(buffer)
    
    if show_progress:
        print(f"   ✓ Created {len(final_chunks)} semantic chunks (from {len(sentences)} sentences)")
    
    return final_chunks


def chunk_text(text: str, mode: ChunkingMode, embeddings=None, 
               chunk_size: int = 500, overlap: int = 50,
               show_progress: bool = False) -> List[str]:
    """
    Universal chunking function that dispatches to the appropriate method.
    
    Args:
        text: The text to chunk
        mode: ChunkingMode (FIXED, SEMANTIC, or SENTENCE)
        embeddings: Required for SEMANTIC mode
        chunk_size: For FIXED mode, the target chunk size
        overlap: For FIXED mode, the overlap between chunks
        show_progress: Whether to show progress for semantic analysis
    """
    if mode == ChunkingMode.FIXED:
        return chunk_text_fixed(text, chunk_size, overlap)
    
    elif mode == ChunkingMode.SENTENCE:
        # Roughly 5 sentences ≈ 500 chars
        return chunk_text_sentences(text, sentences_per_chunk=5, overlap_sentences=1)
    
    elif mode == ChunkingMode.SEMANTIC:
        if embeddings is None:
            print("   ⚠ No embeddings provided for semantic chunking, falling back to sentence mode")
            return chunk_text_sentences(text)
        return chunk_text_semantic(text, embeddings, show_progress=show_progress)
    
    else:
        # Default to fixed
        return chunk_text_fixed(text, chunk_size, overlap)

# ============================================================================
# Vector Database Manager (with progress)
# ============================================================================

class VectorDBManager:
    """Manages the vector database for long-term memory"""
    
    def __init__(self, db_dir: Path = VECTOR_DB_DIR, collection: str = COLLECTION_NAME, 
                 use_lm_studio: bool = False):
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection
        
        # Choose embedding model
        if use_lm_studio:
            print("📡 Using LM Studio embeddings...")
            self.embeddings = LMStudioEmbeddings()
        else:
            print("🤗 Using HuggingFace embeddings...")
            self.embeddings = ProgressHuggingFaceEmbeddings()
        
        # Initialize Chroma
        print("📂 Opening vector database...")
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.db_dir)
        )
        
        print(f"✓ Vector DB ready: {self.db_dir}")
        print_system_stats()
    
    def add_texts_with_progress(self, texts: List[str], metadatas: List[dict] = None, 
                                 batch_size: int = DB_INSERT_BATCH_SIZE) -> int:
        """Add texts to the database with progress display"""
        if not texts:
            return 0
        
        total = len(texts)
        added = 0
        
        print(f"\n💾 Adding {total} documents to database...")
        progress = ProgressBar(total, desc="📥 Inserting", show_speed=True)
        
        try:
            for i in range(0, total, batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_meta = metadatas[i:i + batch_size] if metadatas else None
                
                try:
                    self.vectorstore.add_texts(batch_texts, metadatas=batch_meta)
                    added += len(batch_texts)
                    progress.update(len(batch_texts), suffix=f"Mem: {get_memory_usage()}")
                except Exception as e:
                    print(f"\n⚠ Error adding batch {i//batch_size + 1}: {e}")
                    # Continue with next batch
            
            progress.finish(f"Added {added}/{total} documents")
        except KeyboardInterrupt:
            progress.cancel()
            print(f"   Partial import: {added} documents added before cancellation")
        
        return added
    
    def add_texts(self, texts: List[str], metadatas: List[dict] = None):
        """Add texts - uses progress version for large batches"""
        if len(texts) > 20:
            return self.add_texts_with_progress(texts, metadatas)
        
        try:
            self.vectorstore.add_texts(texts, metadatas=metadatas)
            return len(texts)
        except Exception as e:
            print(f"⚠ Error adding texts: {e}")
            return 0
    
    def search(self, query: str, k: int = 5):
        """Search for similar documents"""
        try:
            return self.vectorstore.similarity_search(query, k=k)
        except Exception as e:
            print(f"⚠ Search error: {e}")
            return []
    
    def search_with_score(self, query: str, k: int = 5):
        """Search with relevance scores"""
        try:
            return self.vectorstore.similarity_search_with_score(query, k=k)
        except Exception as e:
            print(f"⚠ Search error: {e}")
            return []
    
    def get_count(self) -> int:
        """Get total document count"""
        try:
            return self.vectorstore._collection.count()
        except:
            return 0
    
    def delete_all(self):
        """Delete all documents"""
        try:
            self.vectorstore.delete_collection()
            # Reinitialize
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.db_dir)
            )
            return True
        except Exception as e:
            print(f"⚠ Delete error: {e}")
            return False

# ============================================================================
# PDF Processing
# ============================================================================

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from a PDF file"""
    if not PDF_SUPPORT:
        print(f"  ⚠ PDF support not available. Install: pip install pymupdf")
        return ""
    
    try:
        doc = fitz.open(str(pdf_path))
        text_parts = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text_parts.append(page.get_text())
        
        doc.close()
        return "\n\n".join(text_parts)
    
    except Exception as e:
        print(f"  ⚠ Error reading PDF {pdf_path.name}: {e}")
        return ""

# ============================================================================
# File Processing Functions (with progress)
# ============================================================================

def read_file_content(file_path: Path) -> Tuple[str, str]:
    """
    Read content from a file. Returns (content, file_type).
    Supports: txt, md, json, log, pdf
    """
    ext = file_path.suffix.lower()
    
    # PDF handling
    if ext == '.pdf':
        return extract_text_from_pdf(file_path), 'pdf'
    
    # Text file handling
    try:
        content = file_path.read_text(encoding='utf-8')
        return content, ext[1:]  # Remove the dot
    except UnicodeDecodeError:
        try:
            content = file_path.read_text(encoding='latin-1')
            return content, ext[1:]
        except Exception as e:
            print(f"  ⚠ Cannot read: {file_path.name} ({e})")
            return "", ext[1:]


def process_file(file_path: Path, chunk_mode: ChunkingMode, embeddings=None,
                 chunk_size: int = 500) -> List[dict]:
    """Process a single file into chunks with metadata"""
    chunks = []
    
    content, file_type = read_file_content(file_path)
    
    if not content.strip():
        return []
    
    # Chunk the text using the selected mode
    text_chunks = chunk_text(
        content, 
        mode=chunk_mode, 
        embeddings=embeddings,
        chunk_size=chunk_size,
        show_progress=False  # Don't show per-file progress
    )
    
    for i, chunk in enumerate(text_chunks):
        chunks.append({
            'text': chunk,
            'metadata': {
                'source': str(file_path),
                'filename': file_path.name,
                'file_type': file_type,
                'chunk_mode': chunk_mode.value,
                'chunk_index': i,
                'total_chunks': len(text_chunks),
                'char_count': len(chunk),
                'imported_at': datetime.now().isoformat()
            }
        })
    
    return chunks


def process_multiple_files(file_paths: List[Path], chunk_mode: ChunkingMode,
                           embeddings=None, chunk_size: int = 500) -> List[dict]:
    """Process multiple files with progress bar"""
    all_chunks = []
    total_files = len(file_paths)
    
    mode_name = chunk_mode.value.upper()
    print(f"\n📂 Processing {total_files} files with {mode_name} chunking...")
    
    if chunk_mode == ChunkingMode.SEMANTIC:
        print("   ℹ Semantic chunking analyzes text meaning - this takes longer but keeps concepts together")
    
    progress = ProgressBar(total_files, desc="📄 Reading files", show_speed=True)
    
    file_stats = {'success': 0, 'failed': 0, 'skipped': 0}
    
    try:
        for i, file_path in enumerate(file_paths):
            try:
                chunks = process_file(file_path, chunk_mode, embeddings, chunk_size)
                
                if chunks:
                    all_chunks.extend(chunks)
                    file_stats['success'] += 1
                else:
                    file_stats['skipped'] += 1
                
                progress.update(1, suffix=f"{file_path.name[:30]} → {len(chunks)} chunks")
                
            except Exception as e:
                file_stats['failed'] += 1
                progress.update(1, suffix=f"⚠ {file_path.name[:30]} failed")
        
        progress.finish(f"Processed {file_stats['success']} files, {len(all_chunks)} chunks")
        
        if file_stats['failed'] > 0:
            print(f"   ⚠ {file_stats['failed']} files failed to process")
        if file_stats['skipped'] > 0:
            print(f"   ℹ {file_stats['skipped']} files were empty/skipped")
    
    except KeyboardInterrupt:
        progress.cancel()
        print(f"   Partial processing: {len(all_chunks)} chunks from {file_stats['success']} files")
    
    return all_chunks

# ============================================================================
# GUI File/Folder Selection
# ============================================================================

def select_files_gui() -> List[Path]:
    """Open a file dialog to select multiple files"""
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        # Include PDF in the file types
        filetypes = [
            ("All supported", "*.txt *.md *.json *.log *.pdf"),
            ("Text files", "*.txt"),
            ("Markdown files", "*.md"),
            ("JSON files", "*.json"),
            ("PDF files", "*.pdf"),
            ("Log files", "*.log"),
            ("All files", "*.*")
        ]
        
        file_paths = filedialog.askopenfilenames(
            title="Select Files to Import",
            filetypes=filetypes
        )
        
        root.destroy()
        return [Path(p) for p in file_paths]
    
    except ImportError:
        print("⚠ tkinter not available. Please enter paths manually.")
        return []
    except Exception as e:
        print(f"⚠ GUI error: {e}")
        return []

def select_folder_gui() -> Optional[Path]:
    """Open a folder dialog"""
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        folder_path = filedialog.askdirectory(title="Select Folder to Import")
        
        root.destroy()
        return Path(folder_path) if folder_path else None
    
    except ImportError:
        print("⚠ tkinter not available. Please enter path manually.")
        return None
    except Exception as e:
        print(f"⚠ GUI error: {e}")
        return None

def get_files_from_folder(folder: Path, extensions: List[str] = None) -> List[Path]:
    """Get all matching files from a folder (recursively)"""
    if extensions is None:
        extensions = ['.txt', '.md', '.json', '.log', '.pdf']
    
    files = []
    for ext in extensions:
        files.extend(folder.rglob(f"*{ext}"))
    
    return sorted(files)

# ============================================================================
# Import Summary
# ============================================================================

def print_import_summary(start_time: float, files_count: int, chunks_count: int, 
                         added_count: int, db_total: int, chunk_mode: ChunkingMode):
    """Print a detailed summary after import"""
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("📊 IMPORT SUMMARY")
    print("=" * 60)
    print(f"   ⏱  Total time:      {format_time(elapsed)}")
    print(f"   📄 Files processed: {files_count}")
    print(f"   📝 Chunks created:  {chunks_count}")
    print(f"   💾 Documents added: {added_count}")
    print(f"   📚 Database total:  {db_total}")
    print(f"   🔧 Chunking mode:   {chunk_mode.value}")
    
    if chunks_count > 0:
        avg_time = elapsed / chunks_count
        print(f"   ⚡ Avg time/chunk:  {avg_time:.2f}s")
    
    print_system_stats()
    print("=" * 60)

# ============================================================================
# Chunking Mode Selection
# ============================================================================

def select_chunking_mode() -> ChunkingMode:
    """Let user select chunking mode"""
    print("\n🔧 Chunking Mode:")
    print("   [1] FIXED    - Fast, fixed-size chunks (default)")
    print("                  Best for: large documents, general text")
    print("   [2] SEMANTIC - Smart topic-aware chunks (slower)")
    print("                  Best for: rulebooks, reference material, structured content")
    print("   [3] SENTENCE - Sentence-based chunks (middle ground)")
    print("                  Best for: articles, narratives")
    
    choice = input("\nChoice [1]: ").strip() or "1"
    
    if choice == "2":
        return ChunkingMode.SEMANTIC
    elif choice == "3":
        return ChunkingMode.SENTENCE
    else:
        return ChunkingMode.FIXED

# ============================================================================
# Interactive Menu
# ============================================================================

def print_menu():
    """Print the main menu"""
    print("\n" + "=" * 60)
    print("📚 Vector Database Manager v4")
    print("   with Semantic Chunking")
    print("=" * 60)
    print()
    print("  [1] 📂 Select files to import (GUI)")
    print("  [2] 📁 Select folder to import (GUI)")
    print("  [3] ⌨️  Enter file/folder path manually")
    print("  [4] 🔍 Search the database")
    print("  [5] 📊 View database stats")
    print("  [6] 🧪 Test embedding model")
    print("  [7] 🗑️  Delete all data (DANGER)")
    print("  [0] 🚪 Exit")
    print()
    
    if PDF_SUPPORT:
        print("  ✓ PDF support: enabled")
    else:
        print("  ⚠ PDF support: disabled (pip install pymupdf)")
    
    if NUMPY_AVAILABLE:
        print("  ✓ Semantic chunking: enabled")
    else:
        print("  ⚠ Semantic chunking: limited (pip install numpy)")
    print()

def run_import(db: VectorDBManager, files: List[Path], chunk_mode: ChunkingMode, 
               chunk_size: int = 500):
    """Run the import process with full progress tracking"""
    start_time = time.time()
    
    # Phase 1: Process files
    chunks = process_multiple_files(
        files, 
        chunk_mode=chunk_mode,
        embeddings=db.embeddings,
        chunk_size=chunk_size
    )
    
    if not chunks:
        print("\n⚠ No content to import.")
        return
    
    # Phase 2: Add to database (embedding + insertion)
    texts = [c['text'] for c in chunks]
    metadatas = [c['metadata'] for c in chunks]
    
    added = db.add_texts_with_progress(texts, metadatas)
    
    # Print summary
    print_import_summary(
        start_time=start_time,
        files_count=len(files),
        chunks_count=len(chunks),
        added_count=added,
        db_total=db.get_count(),
        chunk_mode=chunk_mode
    )

def main():
    """Main CLI interface"""
    print("\n" + "=" * 60)
    print("Vector Database Manager v4")
    print("With Semantic Chunking & Progress Bars")
    print("=" * 60)
    
    # Choose embedding model
    print("\nEmbedding model:")
    print("  [1] HuggingFace (local, no server needed) [RECOMMENDED]")
    print("  [2] LM Studio (requires LM Studio running with embedding model)")
    
    choice = input("\nChoice [1]: ").strip() or "1"
    use_lm_studio = choice == "2"
    
    # Initialize database
    print()
    db = VectorDBManager(use_lm_studio=use_lm_studio)
    print(f"\n📊 Current documents: {db.get_count()}")
    
    while True:
        print_menu()
        choice = input("Choice: ").strip()
        
        # ----- Import via GUI file picker -----
        if choice == "1":
            print("\n📂 Opening file selector...")
            files = select_files_gui()
            
            if not files:
                print("No files selected.")
                continue
            
            print(f"\nSelected {len(files)} file(s):")
            for f in files[:10]:  # Show first 10
                print(f"  - {f.name}")
            if len(files) > 10:
                print(f"  ... and {len(files) - 10} more")
            
            # Select chunking mode
            chunk_mode = select_chunking_mode()
            
            chunk_size = 500
            if chunk_mode == ChunkingMode.FIXED:
                chunk_size_input = input("\nChunk size [500]: ").strip()
                chunk_size = int(chunk_size_input) if chunk_size_input else 500
            
            run_import(db, files, chunk_mode, chunk_size)
        
        # ----- Import via GUI folder picker -----
        elif choice == "2":
            print("\n📁 Opening folder selector...")
            folder = select_folder_gui()
            
            if not folder:
                print("No folder selected.")
                continue
            
            print(f"\nSelected: {folder}")
            
            # Get file extensions (now includes PDF by default)
            default_ext = ".txt,.md,.json,.log,.pdf" if PDF_SUPPORT else ".txt,.md,.json,.log"
            ext_input = input(f"File extensions [{default_ext}]: ").strip()
            
            if ext_input:
                extensions = [e.strip() if e.startswith('.') else f'.{e.strip()}' 
                             for e in ext_input.split(',')]
            else:
                extensions = default_ext.split(',')
            
            print(f"\n🔍 Scanning for files...")
            files = get_files_from_folder(folder, extensions)
            
            if not files:
                print(f"No files found with extensions: {extensions}")
                continue
            
            print(f"Found {len(files)} file(s)")
            
            # Show file type breakdown
            by_type = {}
            for f in files:
                ext = f.suffix.lower()
                by_type[ext] = by_type.get(ext, 0) + 1
            
            print("   File types: " + ", ".join(f"{ext}: {count}" for ext, count in sorted(by_type.items())))
            
            confirm = input("\nImport all? (y/n) [y]: ").strip().lower()
            if confirm and confirm != 'y':
                continue
            
            # Select chunking mode
            chunk_mode = select_chunking_mode()
            
            chunk_size = 500
            if chunk_mode == ChunkingMode.FIXED:
                chunk_size_input = input("\nChunk size [500]: ").strip()
                chunk_size = int(chunk_size_input) if chunk_size_input else 500
            
            run_import(db, files, chunk_mode, chunk_size)
        
        # ----- Manual path entry -----
        elif choice == "3":
            path_input = input("\nEnter file or folder path: ").strip()
            
            if not path_input:
                continue
            
            # Handle quoted paths
            path_input = path_input.strip('"').strip("'")
            path = Path(path_input)
            
            if not path.exists():
                print(f"⚠ Path not found: {path}")
                continue
            
            if path.is_file():
                files = [path]
            else:
                default_ext = ".txt,.md,.json,.log,.pdf" if PDF_SUPPORT else ".txt,.md,.json,.log"
                ext_input = input(f"File extensions [{default_ext}]: ").strip()
                
                if ext_input:
                    extensions = [e.strip() if e.startswith('.') else f'.{e.strip()}' 
                                 for e in ext_input.split(',')]
                else:
                    extensions = default_ext.split(',')
                
                files = get_files_from_folder(path, extensions)
            
            if not files:
                print("No files found.")
                continue
            
            # Select chunking mode
            chunk_mode = select_chunking_mode()
            
            chunk_size = 500
            if chunk_mode == ChunkingMode.FIXED:
                chunk_size_input = input("\nChunk size [500]: ").strip()
                chunk_size = int(chunk_size_input) if chunk_size_input else 500
            
            run_import(db, files, chunk_mode, chunk_size)
        
        # ----- Search -----
        elif choice == "4":
            query = input("\n🔍 Search query: ").strip()
            if not query:
                continue
            
            k = input("Number of results [5]: ").strip()
            k = int(k) if k else 5
            
            print(f"\nSearching for: '{query}'...")
            
            spinner = SpinnerProgress("Searching")
            spinner.start()
            results = db.search_with_score(query, k=k)
            spinner.stop()
            
            if not results:
                print("No results found.")
                continue
            
            print(f"\n📋 Found {len(results)} result(s):\n")
            for i, (doc, score) in enumerate(results, 1):
                print(f"{'─' * 50}")
                print(f"Result {i} | Score: {score:.4f}")
                if doc.metadata.get('filename'):
                    print(f"Source: {doc.metadata['filename']}")
                if doc.metadata.get('file_type'):
                    print(f"Type: {doc.metadata['file_type']}")
                if doc.metadata.get('chunk_mode'):
                    print(f"Chunked: {doc.metadata['chunk_mode']}")
                print(f"{'─' * 50}")
                preview = doc.page_content[:300]
                if len(doc.page_content) > 300:
                    preview += "..."
                print(preview)
                print()
        
        # ----- Stats -----
        elif choice == "5":
            count = db.get_count()
            print(f"\n📊 Database Statistics")
            print(f"   Documents:  {count}")
            print(f"   Location:   {db.db_dir.resolve()}")
            print(f"   Collection: {db.collection_name}")
            print_system_stats()
        
        # ----- Test embedding -----
        elif choice == "6":
            test_text = "This is a test sentence to verify the embedding model works correctly."
            print(f"\n🧪 Testing: '{test_text}'")
            
            try:
                start = time.time()
                embedding = db.embeddings.embed_query(test_text)
                elapsed = time.time() - start
                
                print(f"✓ Embedding generated in {elapsed:.2f}s")
                print(f"  Dimension: {len(embedding)}")
                print(f"  Sample values: {embedding[:5]}")
            except Exception as e:
                print(f"✗ Error: {e}")
        
        # ----- Delete all -----
        elif choice == "7":
            print("\n⚠️  WARNING: This will delete ALL data!")
            confirm = input("Type 'DELETE' to confirm: ").strip()
            
            if confirm == "DELETE":
                spinner = SpinnerProgress("Deleting")
                spinner.start()
                success = db.delete_all()
                spinner.stop("All data deleted" if success else "Delete failed")
            else:
                print("Cancelled.")
        
        # ----- Exit -----
        elif choice == "0":
            print("\n👋 Goodbye!")
            break
        
        else:
            print("Invalid choice.")

# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
