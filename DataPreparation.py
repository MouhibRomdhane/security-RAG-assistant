
import datetime
from langchain_community.document_loaders import DirectoryLoader
from dotenv import load_dotenv
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
load_dotenv()
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader,UnstructuredMarkdownLoader
import time
from langchain_pymupdf4llm import PyMuPDF4LLMLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
import pickle
import os
from pathlib import Path

file_path =str( Path(os.getenv("datapath")))

print("Setting up Loaders for: PDF (Smart) + Markdown (MITRE/CSF)")

loaders = [
    DirectoryLoader(
        file_path,
        glob="*.pdf",
        loader_cls=PyMuPDF4LLMLoader, # 
        show_progress=True,
        use_multithreading=True
    ),

   DirectoryLoader(
        file_path,
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader,
        loader_kwargs={
            "mode": "single", 
            "autodetect_encoding": True 
        },
        show_progress=True,
        use_multithreading=True,
        max_concurrency=8 
    )

]

print("🚀 Starting Ingestion...")
docs = []
start_time = time.time()

for loader in loaders:
    try:
        new_docs = loader.load()
        if len(new_docs) > 0:
            source_type = "PDF_to_Markdown" if "pdf" in new_docs[0].metadata.get("source", "") else "Markdown_File"
            print(f" oaded {len(new_docs)} files (Type: {source_type})")
            docs.extend(new_docs)
    except Exception as e:
        print(f" Error in loader: {e}")

duration = time.time() - start_time
print(f"🎉 Total Loaded: {len(docs)} documents in {duration:.2f} seconds")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,          
    chunk_overlap=150,       
    length_function=len,
    # Priority: Split by Paragraphs (\n\n) -> Lines (\n) -> Sentences
    separators=["\n\n", "\n", " ", ""] 
)

print("Chunking documents...")
chunks = text_splitter.split_documents(docs)

print(f"Final Database Ready: {len(chunks)} chunks created.")

""""
if len(chunks) > 0:
    print("\n--- Sample Chunk Content ---")
    print(chunks[0].page_content[:500])
    print("\n--- Metadata ---")
    print(chunks[0].metadata)
"""
BM25_PATH = Path(os.getenv("BM25_PATH"))
# 4. BUILD BM25 RETRIEVER (SPARSE INDEX)
print("4. Building BM25 Index (Sparse Index)...")
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 50  # We want a wide net for the first pass
# 5. SAVE BM25 TO DISK
print(f"5. Saving BM25 retriever to {BM25_PATH}...")
with open(BM25_PATH, "wb") as f:
    pickle.dump(bm25_retriever, f)
    
("--- BUILD COMPLETE ---")

print("Loading BGE-M3 Model (This is large, please wait)...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True, "batch_size": 4} 
)

start_time = time.time()
vdb=str(Path(os.getenv("vdb")))
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=vdb
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