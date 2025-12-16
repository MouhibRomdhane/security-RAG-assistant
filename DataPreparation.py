
import datetime

from langchain_community.document_loaders import DirectoryLoader

from dotenv import load_dotenv
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader,PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=600)

import os
from langchain_community.document_loaders import DirectoryLoader,UnstructuredMarkdownLoader

import time

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

duration = time.time() - start_time
print(f"🎉 Total Loaded: {len(docs)} documents in {duration:.2f} seconds")

# 4. Configure the Universal Splitter
# This splitter handles both the "PDF-Markdown" and the "CSF-Text" safely.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,          # 1000 is usually better than 2000 for retrieval precision
    chunk_overlap=150,        # Overlap to keep context across cuts
    length_function=len,
    # Priority: Split by Paragraphs (\n\n) -> Lines (\n) -> Sentences
    separators=["\n\n", "\n", " ", ""] 
)

print("✂️ Chunking documents...")
chunks = text_splitter.split_documents(docs)

print(f"✅ Final Database Ready: {len(chunks)} chunks created.")

# 5. Verification: Check a random chunk to ensure headers/tables look good
if len(chunks) > 0:
    print("\n--- Sample Chunk Content ---")
    print(chunks[0].page_content[:500])
    print("\n--- Metadata ---")
    print(chunks[0].metadata)

# 1. Define the BGE-M3 Model
print("Loading BGE-M3 Model (This is large, please wait)...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cuda"},
    # EXPLICITLY force it to be small
    encode_kwargs={"normalize_embeddings": True, "batch_size": 2} 
)

start_time = time.time()

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./vectordb"
    )
print("data saved")
end_time = time.time()
duration = end_time - start_time

# Calculate stats
chunks_per_sec = len(chunks) / duration
formatted_time = str(datetime.timedelta(seconds=int(duration)))

print("\n" + "="*40)
print(f"✅ FINISHED!")
print(f"⏱️ Total Time:     {formatted_time} (Hours:Minutes:Seconds)")
print(f"🚀 Speed:          {chunks_per_sec:.2f} chunks/second")
print("="*40)