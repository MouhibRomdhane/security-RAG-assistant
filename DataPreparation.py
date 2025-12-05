import os
os.environ['HF_HOME'] = "D:\\huggingface_cache"
from langchain_community.document_loaders import DirectoryLoader

from dotenv import load_dotenv
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader,PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_unstructured import UnstructuredLoader
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=600)

import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader,JSONLoader

file_path = "./Data"

loaders = [
    # 1. PDF Loader
    # Note: We removed 'mode' and 'strategy' because PyPDFLoader doesn't use them.
    DirectoryLoader(
        file_path, 
        glob="**/*.pdf", 
        loader_cls=PDFMinerLoader,
        show_progress=True,          # <--- Adds the progress bar
        use_multithreading=True,     # <--- Speeds up loading
        silent_errors=True
    ),
    
    # 2. Text/JSON Loader
    # We add 'autodetect_encoding' to loader_kwargs to prevent crashing on weird characters
    DirectoryLoader(
        file_path, 
        glob="**/*.json", 
        loader_cls=JSONLoader,
        loader_kwargs={
            # This schema says: "Go inside 'objects', grab EVERYTHING that isn't a relationship, 
            # and combine the Type, Name, and Description into one text block."
            "jq_schema": '.objects[] | select(.type != "relationship") | "Type: " + .type + "\nName: " + .name + "\nDescription: " + (.description // "No description")', 
            "text_content": False
        },
        show_progress=True
    ),
    # 3. Text Files
    DirectoryLoader(
        file_path, 
        glob="**/*.txt", 
        loader_cls=TextLoader,
        show_progress=True,
        loader_kwargs={"autodetect_encoding": True}
    )
]

print("🔄 Loading documents...")
docs = []
for loader in loaders:
    try:
        # The progress bar will appear here automatically
        new_docs = loader.load()
        docs.extend(new_docs)
        print(f"\n   ✅ Loaded {len(new_docs)} files from {loader.loader_cls.__name__}")
    except Exception as e:
        print(f"\n   ❌ Error in loader: {e}")

print(f"🎉 Total Loaded: {len(docs)} documents")
# 1. Configure the Splitter for Incident Response
# We use a large chunk_size (2000) to keep full procedures intact.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,     # Large window for full context
    chunk_overlap=250,   # High overlap to bridge page transitions
    length_function=len,
    is_separator_regex=False,
    # Priority: Split by paragraphs (\n\n) first, then lines, then sentences
    separators=["\n\n", "\n", ". ", " ", ""]
)

# 2. Apply the splitting
print("✂️ Chunking documents...")
chunks = text_splitter.split_documents(docs)

# 3. Verification
print(f"✅ Original Documents: {len(docs)}")
print(f"📦 Created Chunks: {len(chunks)}")

# 4. (Optional) Inspection - See what a chunk looks like
print("\n--- Sample Chunk (Content) ---")
print(chunks[0].page_content[:500] + "...")
print("\n--- Sample Chunk (Metadata) ---")
print(chunks[0].metadata)

"""" 
embeddings = HuggingFaceEmbeddings(
model_name="BAAI/bge-m3"
)
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./vectordb"
    )
print("data saved")"""