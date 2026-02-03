"""
Vector Database Manager v2
- Easy file/folder selection via GUI
- Modern LangChain imports (no deprecation warnings)
- Support for LM Studio embeddings or HuggingFace
- Batch processing with progress display
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
VECTOR_DB_DIR = Path("./vector_db")
COLLECTION_NAME = "long_term_memory"

# ============================================================================
# Modern LangChain Imports
# ============================================================================

try:
    from langchain_chroma import Chroma
except ImportError:
    print("[!] Missing: pip install langchain-chroma chromadb")
    sys.exit(1)

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    print("[!] Missing: pip install langchain-huggingface")
    sys.exit(1)

from langchain_core.embeddings import Embeddings

# ============================================================================
# LM Studio Embedding Class
# ============================================================================

class LMStudioEmbeddings(Embeddings):
    """
    Custom embedding class that uses LM Studio's API.
    Requires an embedding model loaded in LM Studio.
    """
    
    def __init__(self, base_url: str = LM_STUDIO_BASE_URL):
        self.base_url = base_url
        self.endpoint = f"{base_url}/embeddings"
        self._dimension = None
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        import requests
        
        try:
            response = requests.post(
                self.endpoint,
                headers={"Content-Type": "application/json"},
                json={"input": text, "model": "embedding-model"},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                embedding = data['data'][0]['embedding']
                self._dimension = len(embedding)
                return embedding
            else:
                print(f"⚠ LM Studio error: {response.status_code}")
                return [0.0] * (self._dimension or 384)
                
        except requests.exceptions.ConnectionError:
            print("⚠ Cannot connect to LM Studio. Is it running?")
            return [0.0] * (self._dimension or 384)
        except Exception as e:
            print(f"⚠ Embedding error: {e}")
            return [0.0] * (self._dimension or 384)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        embeddings = []
        for i, text in enumerate(texts):
            if (i + 1) % 10 == 0:
                print(f"  Embedding {i + 1}/{len(texts)}...")
            embeddings.append(self._get_embedding(text))
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query string"""
        return self._get_embedding(text)

# ============================================================================
# Vector Database Manager
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
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        
        # Initialize Chroma
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.db_dir)
        )
        
        print(f"✓ Vector DB ready: {self.db_dir}")
    
    def add_texts(self, texts: List[str], metadatas: List[dict] = None):
        """Add texts to the database"""
        if not texts:
            return 0
        
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
# File Processing Functions
# ============================================================================

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence/paragraph boundary
        if end < len(text):
            # Look for newline or period near the end
            for sep in ['\n\n', '\n', '. ', '! ', '? ']:
                idx = chunk.rfind(sep)
                if idx > chunk_size * 0.5:  # Only if past halfway
                    chunk = chunk[:idx + len(sep)]
                    end = start + len(chunk)
                    break
        
        if chunk.strip():
            chunks.append(chunk.strip())
        
        start = end - overlap
    
    return chunks

def process_file(file_path: Path, chunk_size: int = 500) -> List[dict]:
    """Process a single file into chunks with metadata"""
    chunks = []
    
    try:
        content = file_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        try:
            content = file_path.read_text(encoding='latin-1')
        except:
            print(f"  ⚠ Cannot read: {file_path.name}")
            return []
    
    text_chunks = chunk_text(content, chunk_size)
    
    for i, chunk in enumerate(text_chunks):
        chunks.append({
            'text': chunk,
            'metadata': {
                'source': str(file_path),
                'filename': file_path.name,
                'chunk_index': i,
                'total_chunks': len(text_chunks),
                'imported_at': datetime.now().isoformat()
            }
        })
    
    return chunks

def process_multiple_files(file_paths: List[Path], chunk_size: int = 500) -> List[dict]:
    """Process multiple files"""
    all_chunks = []
    
    for i, file_path in enumerate(file_paths):
        print(f"  [{i+1}/{len(file_paths)}] {file_path.name}...")
        chunks = process_file(file_path, chunk_size)
        all_chunks.extend(chunks)
        print(f"    → {len(chunks)} chunks")
    
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
        root.withdraw()  # Hide the main window
        root.attributes('-topmost', True)  # Bring dialog to front
        
        file_paths = filedialog.askopenfilenames(
            title="Select Files to Import",
            filetypes=[
                ("Text files", "*.txt"),
                ("Markdown files", "*.md"),
                ("JSON files", "*.json"),
                ("Log files", "*.log"),
                ("All files", "*.*")
            ]
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
        extensions = ['.txt', '.md', '.json', '.log']
    
    files = []
    for ext in extensions:
        files.extend(folder.rglob(f"*{ext}"))
    
    return sorted(files)

# ============================================================================
# Interactive Menu
# ============================================================================

def print_menu():
    """Print the main menu"""
    print("\n" + "=" * 60)
    print("📚 Vector Database Manager v2")
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

def main():
    """Main CLI interface"""
    print("\n" + "=" * 60)
    print("Vector Database Manager v2")
    print("=" * 60)
    
    # Choose embedding model
    print("\nEmbedding model:")
    print("  [1] HuggingFace (local, no server needed)")
    print("  [2] LM Studio (requires LM Studio running)")
    
    choice = input("\nChoice [1]: ").strip() or "1"
    use_lm_studio = choice == "2"
    
    # Initialize database
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
            for f in files:
                print(f"  - {f.name}")
            
            chunk_size = input("\nChunk size [500]: ").strip()
            chunk_size = int(chunk_size) if chunk_size else 500
            
            print(f"\n📝 Processing {len(files)} files...")
            chunks = process_multiple_files(files, chunk_size)
            
            if chunks:
                print(f"\n💾 Importing {len(chunks)} chunks...")
                texts = [c['text'] for c in chunks]
                metadatas = [c['metadata'] for c in chunks]
                added = db.add_texts(texts, metadatas)
                print(f"✓ Added {added} documents. Total: {db.get_count()}")
        
        # ----- Import via GUI folder picker -----
        elif choice == "2":
            print("\n📁 Opening folder selector...")
            folder = select_folder_gui()
            
            if not folder:
                print("No folder selected.")
                continue
            
            print(f"\nSelected: {folder}")
            
            # Get file extensions
            ext_input = input("File extensions [.txt,.md,.json,.log]: ").strip()
            if ext_input:
                extensions = [e.strip() if e.startswith('.') else f'.{e.strip()}' 
                             for e in ext_input.split(',')]
            else:
                extensions = ['.txt', '.md', '.json', '.log']
            
            files = get_files_from_folder(folder, extensions)
            
            if not files:
                print(f"No files found with extensions: {extensions}")
                continue
            
            print(f"\nFound {len(files)} file(s)")
            
            confirm = input("Import all? (y/n) [y]: ").strip().lower()
            if confirm and confirm != 'y':
                continue
            
            chunk_size = input("Chunk size [500]: ").strip()
            chunk_size = int(chunk_size) if chunk_size else 500
            
            print(f"\n📝 Processing {len(files)} files...")
            chunks = process_multiple_files(files, chunk_size)
            
            if chunks:
                print(f"\n💾 Importing {len(chunks)} chunks...")
                texts = [c['text'] for c in chunks]
                metadatas = [c['metadata'] for c in chunks]
                added = db.add_texts(texts, metadatas)
                print(f"✓ Added {added} documents. Total: {db.get_count()}")
        
        # ----- Manual path entry -----
        elif choice == "3":
            path_input = input("\nEnter file or folder path: ").strip()
            
            if not path_input:
                continue
            
            path = Path(path_input)
            
            if not path.exists():
                print(f"⚠ Path not found: {path}")
                continue
            
            chunk_size = input("Chunk size [500]: ").strip()
            chunk_size = int(chunk_size) if chunk_size else 500
            
            if path.is_file():
                files = [path]
            else:
                ext_input = input("File extensions [.txt,.md,.json,.log]: ").strip()
                if ext_input:
                    extensions = [e.strip() if e.startswith('.') else f'.{e.strip()}' 
                                 for e in ext_input.split(',')]
                else:
                    extensions = ['.txt', '.md', '.json', '.log']
                files = get_files_from_folder(path, extensions)
            
            if not files:
                print("No files found.")
                continue
            
            print(f"\n📝 Processing {len(files)} file(s)...")
            chunks = process_multiple_files(files, chunk_size)
            
            if chunks:
                print(f"\n💾 Importing {len(chunks)} chunks...")
                texts = [c['text'] for c in chunks]
                metadatas = [c['metadata'] for c in chunks]
                added = db.add_texts(texts, metadatas)
                print(f"✓ Added {added} documents. Total: {db.get_count()}")
        
        # ----- Search -----
        elif choice == "4":
            query = input("\n🔍 Search query: ").strip()
            if not query:
                continue
            
            k = input("Number of results [5]: ").strip()
            k = int(k) if k else 5
            
            print(f"\nSearching for: '{query}'...\n")
            results = db.search_with_score(query, k=k)
            
            if not results:
                print("No results found.")
                continue
            
            print(f"Found {len(results)} result(s):\n")
            for i, (doc, score) in enumerate(results, 1):
                print(f"{'─' * 50}")
                print(f"Result {i} | Score: {score:.4f}")
                if doc.metadata.get('filename'):
                    print(f"Source: {doc.metadata['filename']}")
                print(f"{'─' * 50}")
                # Show preview
                preview = doc.page_content[:300]
                if len(doc.page_content) > 300:
                    preview += "..."
                print(preview)
                print()
        
        # ----- Stats -----
        elif choice == "5":
            count = db.get_count()
            print(f"\n📊 Database Statistics")
            print(f"   Documents: {count}")
            print(f"   Location: {db.db_dir.resolve()}")
            print(f"   Collection: {db.collection_name}")
        
        # ----- Test embedding -----
        elif choice == "6":
            test_text = "This is a test sentence to verify the embedding model works correctly."
            print(f"\n🧪 Testing: '{test_text}'")
            
            try:
                embedding = db.embeddings.embed_query(test_text)
                print(f"✓ Embedding generated!")
                print(f"  Dimension: {len(embedding)}")
                print(f"  Sample values: {embedding[:5]}")
            except Exception as e:
                print(f"✗ Error: {e}")
        
        # ----- Delete all -----
        elif choice == "7":
            print("\n⚠️  WARNING: This will delete ALL data!")
            confirm = input("Type 'DELETE' to confirm: ").strip()
            
            if confirm == "DELETE":
                if db.delete_all():
                    print("✓ All data deleted.")
                else:
                    print("⚠ Delete failed.")
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
