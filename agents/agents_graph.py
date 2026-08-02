import os
import toml
import streamlit as st
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from utils.rag_engine import search_aircargo_rules

# Define the State for LangGraph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query_category: str
    research_data: str
    critic_feedback: str
    final_report: str

def get_api_keys():
    """Safely get API keys from Streamlit secrets or local toml."""
    try:
        groq_key = st.secrets["GROQ_API_KEY"]
        or_key = st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        secrets_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".streamlit", "secrets.toml")
        if os.path.exists(secrets_path):
            secrets = toml.load(secrets_path)
            groq_key = secrets.get("GROQ_API_KEY", "")
            or_key = secrets.get("OPENROUTER_API_KEY", "")
        else:
            groq_key, or_key = "", ""
    return groq_key, or_key

GROQ_KEY, OR_KEY = get_api_keys()

# Initialize Models (Adhering to the Free Model requirement)
# 1. Router Model: Ultra-low latency via Groq
router_llm = ChatGroq(
    api_key=GROQ_KEY,
    model="llama-3.1-8b-instant",
    temperature=0.0
)

# 2. Researcher Model: 32k context for RAG via Groq
researcher_llm = ChatGroq(
    api_key=GROQ_KEY,
    model="llama-3.3-70b-versatile",
    temperature=0.2
)

# 3. Critic Model: High-reasoning model via Groq (Switched from OpenRouter due to unstable free tier)
critic_llm = ChatGroq(
    api_key=GROQ_KEY,
    model="llama-3.3-70b-versatile",
    temperature=0.3
)

# --- Agent Functions (Nodes) ---

def router_agent(state: AgentState):
    """
    Pattern 1: Router Pattern
    Categorizes the user query.
    """
    last_message = state["messages"][-1].content
    
    prompt = f"""
    Categorize the following user query about Sri Lankan Air Cargo into exactly one of these categories:
    [RATES_AND_WEIGHTS, DANGEROUS_GOODS, EXPORT_COMPLIANCE]
    
    Query: {last_message}
    
    Output ONLY the category name.
    """
    response = router_llm.invoke([HumanMessage(content=prompt)])
    category = response.content.strip()
    
    # Fallback validation
    if category not in ["RATES_AND_WEIGHTS", "DANGEROUS_GOODS", "EXPORT_COMPLIANCE"]:
        category = "EXPORT_COMPLIANCE"
        
    return {"query_category": category}

def researcher_agent(state: AgentState):
    """
    Pattern 2: Tool-Use / RAG Pattern
    Uses the ChromaDB vector store tool to find relevant data.
    """
    last_message = state["messages"][-1].content
    category = state["query_category"]
    
    # Execute Tool
    rag_results = search_aircargo_rules(f"{category} Guidelines: {last_message}", k=5)
    
    # Synthesize findings
    prompt = f"""
    You are a Logistics Researcher Agent for SriLankan Air Cargo.
    User Query: {last_message}
    Category: {category}
    
    Relevant Documents Found:
    {rag_results}
    
    Please provide a comprehensive summary of the rules and regulations that apply to this query based ONLY on the documents provided.
    """
    
    response = researcher_llm.invoke([HumanMessage(content=prompt)])
    return {"research_data": response.content}

def critic_agent(state: AgentState):
    """
    Pattern 3: Reflection / Self-Critique Pattern
    Inspects the researcher's findings against safety standards.
    """
    last_message = state["messages"][-1].content
    research = state["research_data"]
    
    prompt = f"""
    You are the Compliance Critic Agent for SriLankan Aviation and Customs standards.
    The Logistics Researcher has provided the following findings for the user query: "{last_message}"
    
    Researcher's Findings:
    {research}
    
    Task:
    Review the researcher's findings and provide a simple, clear, and highly accurate answer directly to the user.
    Do not generate a massive report or checklist unless explicitly requested by the user. Just give the exact information they need concisely and professionally.
    """
    
    response = critic_llm.invoke([HumanMessage(content=prompt)])
    
    # We append the final report to messages so it can be returned
    return {
        "critic_feedback": "Critique complete.", 
        "final_report": response.content,
        "messages": [AIMessage(content=response.content)]
    }

# --- Graph Definition ---
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("router", router_agent)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("critic", critic_agent)

# Add Edges (Sequential Flow)
workflow.add_edge(START, "router")
workflow.add_edge("router", "researcher")
workflow.add_edge("researcher", "critic")
workflow.add_edge("critic", END)

# Compile Graph
aircargo_advisor_app = workflow.compile()

def run_aircargo_advisor(query: str) -> str:
    """Entry point for the Streamlit UI to call the agent graph."""
    initial_state = {"messages": [HumanMessage(content=query)]}
    final_state = aircargo_advisor_app.invoke(initial_state)
    return final_state["messages"][-1].content
