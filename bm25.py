import os
import datetime
os.environ['HF_HOME'] = "D:\\huggingface_cache"
from langchain_community.document_loaders import DirectoryLoader

from dotenv import load_dotenv
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader,PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=600)

import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader,JSONLoader,UnstructuredMarkdownLoader

import time
from langchain_community.document_loaders import DirectoryLoader, TextLoader, JSONLoader
from langchain_pymupdf4llm import PyMuPDF4LLMLoader # <--- The Pro PDF Loader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Path to your data (From your previous screenshots)
file_path = r".\data"

print("🔄 Setting up Loaders for: PDF (Smart) + Markdown (MITRE/CSF)...")

# We define a custom list of loaders to handle different file types differently
loaders = [
    # 1. PDF Loader (The "Pro" Switch)
    # converting PDFs to Markdown first for better tables/headers
    DirectoryLoader(
        file_path,
        glob="*.pdf",
        loader_cls=PyMuPDF4LLMLoader, # <--- Converts PDF to Markdown structure
        show_progress=True,
        use_multithreading=True
    ),

    # 2. Markdown Loader (For MITRE & CSF folders)
    # We use explicit TextLoader because your screenshots showed .md files
    DirectoryLoader(
        file_path,
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader,
        loader_kwargs={
            "mode": "single", # <--- CRITICAL: Keeps file as one piece. 'elements' is too slow.
            "autodetect_encoding": True 
        },
        show_progress=True,
        use_multithreading=True,
        max_concurrency=8 # Don't go too high with Unstructured, it eats CPU
    ),

    # 3. (Optional) JSON Loader
    # Only uncomment this if you actually have STIX JSON files. 
    # Since you have the Markdown folders, you likely do NOT need this.
    # DirectoryLoader(
    #     file_path, 
    #     glob="**/*.json", 
    #     loader_cls=JSONLoader,
    #     loader_kwargs={
    #         "jq_schema": '.objects[] | select(.type != "relationship") | "Type: " + .type + "\nName: " + .name + "\nDescription: " + (.description // "No description")', 
    #         "text_content": False
    #     },
    #     show_progress=True
    # ),
]

print("🚀 Starting Ingestion...")
docs = []
start_time = time.time()

for loader in loaders:
    try:
        new_docs = loader.load()
        # Tag them so you know where they came from later
        if len(new_docs) > 0:
            source_type = "PDF_to_Markdown" if "pdf" in new_docs[0].metadata.get("source", "") else "Markdown_File"
            print(f"   ✅ Loaded {len(new_docs)} files (Type: {source_type})")
            docs.extend(new_docs)
    except Exception as e:
        print(f"   ❌ Error in loader: {e}")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,          # 1000 is usually better than 2000 for retrieval precision
    chunk_overlap=150,        # Overlap to keep context across cuts
    length_function=len,
    # Priority: Split by Paragraphs (\n\n) -> Lines (\n) -> Sentences
    separators=["\n\n", "\n", " ", ""] 
)

print("✂️ Chunking documents...")
chunks = text_splitter.split_documents(docs)
from langchain_community.retrievers import BM25Retriever
import pickle
BM25_PATH = "bm25_retriever.pkl"
# 4. BUILD BM25 RETRIEVER (SPARSE INDEX)
print("4. Building BM25 Index (Sparse Index)...")
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 50  # We want a wide net for the first pass
# 5. SAVE BM25 TO DISK
print(f"5. Saving BM25 retriever to {BM25_PATH}...")
with open(BM25_PATH, "wb") as f:
    pickle.dump(bm25_retriever, f)
    
("--- BUILD COMPLETE ---")