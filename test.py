from dotenv import load_dotenv
load_dotenv()
import os
os.environ['HF_HOME'] = "D:\\huggingface_cache"
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END, MessagesState,START
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
    retriv_result: Optional[List[str]] = None

vectorstore = Chroma(persist_directory="./vectordb", embedding_function=embeddings)

def retrivinfo(state:AssisstantState):
    last_message = state["messages"][-1].content
    retrieved_docs = vectorstore.similarity_search(last_message,k=6)
    state["retriv_result"] = [doc.page_content for doc in retrieved_docs]
    return state



from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser




def generate_response(state: AssisstantState):
    conversation = state["messages"][:-1]
    last_message = state["messages"][-1].content

    context = "\n\n".join(state["retriv_result"] or ["No relevant information found."])
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

# Create workflow connections
workflow.add_edge(START, "retrivinfo")
workflow.add_edge("retrivinfo", "generate_response")
workflow.add_edge("generate_response", END)
graph = workflow.compile()