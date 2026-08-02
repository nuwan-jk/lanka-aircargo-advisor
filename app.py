import streamlit as st
import os
from utils.ocr_parser import extract_text_from_image_or_pdf
from agents.agents_graph import run_aircargo_advisor

# Configure Streamlit page
st.set_page_config(
    page_title="LankaAir Cargo & Export Compliance Advisor",
    page_icon="✈️",
    layout="wide"
)

# --- Sidebar ---
with st.sidebar:
    st.title("✈️ LankaAir AI Advisor")
    st.markdown("---")
    
    st.markdown("### 🧠 Agentic Design Patterns")
    st.info("""
    **1. Router Pattern**
    Categorizes user queries into specific topics (e.g., Export Compliance, Rates) for specialized processing.
    
    **2. Tool-Use / RAG Pattern**
    A Logistics Researcher agent equipped with ChromaDB to query 20+ Sri Lankan aviation/customs documents.
    
    **3. Reflection / Self-Critique**
    A Compliance Critic agent that reviews the researcher's findings against actual regulations before finalizing.
    """)
    
    st.markdown("### 🤖 Model Selection (Free Tier)")
    st.table({
        "Role": ["Router", "Researcher (RAG)", "Critic", "OCR Scanner"],
        "Model": [
            "llama-3.1-8b (Groq)",
            "mixtral-8x7b (Groq)",
            "gemini-2.0-flash (OpenRouter)",
            "llama-3.2-11b-vision (Groq)"
        ],
        "Reasoning": [
            "Ultra-low latency",
            "32k context for docs",
            "High reasoning/critique",
            "Free vision parsing"
        ]
    })
    st.markdown("---")
    st.caption("Developed for Sri Lankan SME Logistics")

# --- Main App ---
st.title("LankaAir Cargo & Export Compliance Advisor")
st.markdown("Welcome! I can help you understand Sri Lankan Air Cargo rules, Customs regulations, and CAASL standards. You can chat with me or upload an invoice/airway bill for AI data extraction.")

# Tabs for different functionalities
tab1, tab2 = st.tabs(["💬 Chat with Agent", "📄 OCR Scanner"])

with tab2:
    st.header("Invoice & Airway Bill OCR Scanner")
    st.markdown("Upload a scanned label or invoice (.png, .jpg, .pdf) to instantly extract product names, weights, dimensions, and destinations.")
    
    uploaded_file = st.file_uploader("Upload Document", type=["png", "jpg", "jpeg", "pdf"])
    
    if uploaded_file is not None:
        if st.button("Extract Data"):
            with st.spinner("Analyzing document with Vision AI..."):
                extracted_data = extract_text_from_image_or_pdf(uploaded_file)
                st.success("Extraction Complete!")
                
                st.markdown("### Extracted Information:")
                st.code(extracted_data, language="markdown")
                
                # Option to feed this into the chat
                if st.button("Send to Advisor for Compliance Check"):
                    st.session_state.messages.append({
                        "role": "user",
                        "content": f"I just scanned this document. Please check if this cargo complies with Sri Lankan export rules:\n{extracted_data}"
                    })
                    st.switch_page("app.py") # Simple refresh to show in chat

with tab1:
    st.header("Logistics Compliance Chat")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask about cargo rules, customs, or dangerous goods..."):
        # Display user message
        st.chat_message("user").markdown(prompt)
        # Add user message to state
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display assistant thinking...
        with st.chat_message("assistant"):
            with st.spinner("Agents are analyzing (Routing -> Researching -> Critiquing)..."):
                try:
                    response = run_aircargo_advisor(prompt)
                    st.markdown(response)
                    # Add assistant response to state
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Agent Orchestration Error: {str(e)}")
