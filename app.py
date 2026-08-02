import streamlit as st
import os
from agents.agents_graph import run_aircargo_advisor

# Configure Streamlit page
st.set_page_config(
    page_title="LankaAir Cargo & Export Compliance Advisor",
    page_icon="✈️",
    layout="wide"
)

# --- Main App ---
st.title("LankaAir Cargo & Export Compliance Advisor")
st.markdown("Welcome! I can help you understand Sri Lankan Air Cargo rules, Customs regulations, and CAASL standards.")

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
