from dotenv import load_dotenv
load_dotenv()
from langgraph.checkpoint.memory import MemorySaver
import pickle
from langchain_core.documents import Document
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import EnsembleRetriever
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END, MessagesState,START
from typing import List,TypedDict, Dict, Optional, Any
from langchain_ollama import ChatOllama
import os
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid

BM25_PATH = Path(os.getenv("BM25_PATH"))
bge_path=Path(os.getenv("BGE_MODEL_PATH"))
rerankerpath=Path(os.getenv("RERANKER_PATH"))
embeddings = HuggingFaceEmbeddings(
    model_name=str(bge_path),  
    model_kwargs={"local_files_only": True, "device": "cpu"},  # Enforce offline mode
    encode_kwargs={"normalize_embeddings": True},              
)

llm = ChatOllama(
    model="gemma:2b",
    temperature=0,
    
)

"""llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    # other params...
) """
class AssisstantState(MessagesState):
    retriv_result: Optional[List[Document]] = None
    reranked_docs: Optional[List[str]] = None
vdb=Path(os.getenv("vdb"))
vectorstore = Chroma(persist_directory=str(vdb), embedding_function=embeddings)
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




def reranking(state:AssisstantState):
    last_message = state["messages"][-1].content
    cross_encoder = HuggingFaceCrossEncoder(model_name=str(rerankerpath),model_kwargs={"device": "cuda","local_files_only": True})
    
  


    compressor = CrossEncoderReranker(
        model=cross_encoder,
        top_n=5
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
SystemMessage(content=f"""You are a cybersecurity incident response assistant operating in a fully offline environment.

Your role:
- Assist security analysts during incident investigation and response
- Provide accurate, structured, and actionable guidance
- Rely ONLY on the retrieved internal documentation provided to you
- Follow industry best practices 

STRICT RULES:
- Do NOT use external knowledge
- Do NOT guess or hallucinate missing information
- If the retrieved context is insufficient, say: 
  "The provided documentation does not contain enough information to answer this question."
- Clearly distinguish facts from recommendations
- Use clear technical language suitable for cybersecurity professionals


Context retrived:
{context}
conversation history:
"""),
*conversation,
HumanMessage(content=f"user's message: {last_message}")
    ])
    
    chain = prompt|llm|StrOutputParser()
    result=chain.invoke({})

    # Add AI message to state
    state["messages"].append(AIMessage(content=result))
    return state

# Add all nodes
memory = MemorySaver()
workflow = StateGraph(AssisstantState)

workflow.add_node("retrivinfo", retrivinfo)
workflow.add_node("generate_response", generate_response)
workflow.add_node("reranking", reranking)
# Create workflow connections
workflow.add_edge(START, "retrivinfo")
workflow.add_edge("retrivinfo", "reranking")
workflow.add_edge("reranking", "generate_response")
workflow.add_edge("generate_response", END)
graph = workflow.compile(checkpointer=memory)





app = FastAPI(title="Chatbot API")



sessions: Dict[str, Dict[str, Any]] = {}

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None  

class ChatResponse(BaseModel):
    session_id: str
    response: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat(request: ChatRequest):
    
    session_id = request.session_id or str(uuid.uuid4())
    
    
    if session_id not in sessions:
        sessions[session_id] = {"active": True}
    
    
    config = {"configurable": {"thread_id": session_id}}
    
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=request.message)]},
        config=config
    )
    
    
    ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
    response = ai_messages[-1].content if ai_messages else "I'm sorry, I couldn't process your request."
    
    return ChatResponse(
        session_id=session_id,
        response=response)
    