from dotenv import load_dotenv
load_dotenv()

import pickle
from langchain_core.documents import Document
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import EnsembleRetriever
import os
os.environ['HF_HOME'] = "D:\\huggingface_cache"
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END, MessagesState,START

BM25_PATH = "bm25_retriever.pkl"
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3"
)
from typing import List,TypedDict, Optional
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    # other params...
) 
class AssisstantState(MessagesState):
    retriv_result: Optional[List[Document]] = None
    reranked_docs: Optional[List[str]] = None

vectorstore = Chroma(persist_directory="./vectordb", embedding_function=embeddings)
with open(BM25_PATH, "rb") as f:
    bm25_retriever = pickle.load(f)
    bm25_retriever.k=10
chroma_retriever=vectorstore.as_retriever(search_kwargs={"k": 10})
hybrid_retriever = EnsembleRetriever(
    retrievers=[chroma_retriever, bm25_retriever],
    weights=[0.5, 0.5] # Adjustable: 0.5 Dense / 0.5 Sparse
)
def retrivinfo(state:AssisstantState):
    last_message = state["messages"][-1].content
    retrieved_docs = hybrid_retriever.invoke(last_message)
    state["retriv_result"] = [doc for doc in retrieved_docs]
    return state


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

def reranking(state:AssisstantState):
    last_message = state["messages"][-1].content
    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L6-v2",model_kwargs={"device": "cuda"})
    
    
    print("✅ Model loaded in Half-Precision (FP16) mode.")


    compressor = CrossEncoderReranker(
        model=cross_encoder,
        top_n=2
    )
    
    reranked = compressor.compress_documents(
        documents=state["retriv_result"],
        query=last_message
    )
    state["reranked_docs"] = [doc.page_content for doc in reranked]
    return state


def generate_response(state: AssisstantState):
    conversation = state["messages"][:-1]
    last_message = state["messages"][-1].content

    context = state["reranked_docs"] 
    prompt = ChatPromptTemplate.from_messages([
SystemMessage(content=f"""You are a Senior Security Incident Response Analyst (Tier 2). 
Your job is to analyze the provided context (logs, playbooks, or past incidents) and provide a remediation strategy.

STRICT RULES:
1. BASE YOUR ANSWER ONLY ON THE CONTEXT. 
2. BE ACTION-ORIENTED. Provide specific commands or steps.
3. CITE SOURCES. Mention which document or log entry supports your claim.
4. If the user provides a raw log, analyze it for Indicators of Compromise (IoCs).

Context provided:
{context}
"""),
*conversation,
HumanMessage(content=last_message)
    ])
    
    chain = prompt|llm|StrOutputParser()
    result=chain.invoke({})

    # Add AI message to state
    state["messages"].append(AIMessage(content=result))
    return state

# Add all nodes

workflow = StateGraph(AssisstantState)

workflow.add_node("retrivinfo", retrivinfo)
workflow.add_node("generate_response", generate_response)
workflow.add_node("reranking", reranking)
# Create workflow connections
workflow.add_edge(START, "retrivinfo")
workflow.add_edge("retrivinfo", "reranking")
workflow.add_edge("reranking", "generate_response")
workflow.add_edge("generate_response", END)
graph = workflow.compile()