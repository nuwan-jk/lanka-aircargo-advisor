import base64
import json
import io
import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from PIL import Image

def get_vision_llm():
    """Initialize the Groq Vision model."""
    # We use llama-3.2-11b-vision-preview as it is free on Groq
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        # Fallback if secrets are not yet loaded properly
        import toml
        import os
        secrets_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".streamlit", "secrets.toml")
        if os.path.exists(secrets_path):
            secrets = toml.load(secrets_path)
            api_key = secrets.get("GROQ_API_KEY", "")
        else:
            api_key = ""
            
    return ChatGroq(
        api_key=api_key,
        model="llama-3.2-11b-vision-preview",
        temperature=0.1
    )

def extract_text_from_image_or_pdf(uploaded_file) -> str:
    """
    Accepts a Streamlit UploadedFile object (.png, .jpg, or .pdf).
    Converts to Base64 and invokes Vision LLM to extract data.
    """
    if uploaded_file is None:
        return "No file provided."

    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    # Read bytes
    file_bytes = uploaded_file.getvalue()
    
    # If it's a PDF, we might need a different approach or we extract text directly if it's text-based.
    # For true OCR on PDF via Vision LLM, we'd normally render it to an image first.
    # For simplicity without complex external dependencies like Poppler, if it's PDF, 
    # we'll try to extract text using pypdf, and if it's an image, we use Vision LLM.
    if file_extension == 'pdf':
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(file_bytes))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            # Since the user requested Vision LLM specifically, we still use an LLM to parse the extracted text
            # into the required format.
            llm = get_vision_llm()
            msg = HumanMessage(
                content=f"You are a logistics data extractor. Extract cargo product names, weights, dimensions, and destination countries from this text extracted from a PDF invoice/airway bill. Format clearly.\n\n{text}"
            )
            response = llm.invoke([msg])
            return response.content
            
        except Exception as e:
            return f"Error parsing PDF: {e}"
            
    elif file_extension in ['png', 'jpg', 'jpeg']:
        try:
            # Convert to base64
            base64_image = base64.b64encode(file_bytes).decode('utf-8')
            
            # Prepare prompt
            prompt = """
            You are a logistics data extractor. 
            Analyze this scanned shipping label or invoice and extract the following information:
            1. Cargo product names/descriptions
            2. Weights (Gross/Net)
            3. Dimensions
            4. Destination Country
            
            Present the extracted data in a clear, structured format. If any field is missing, state 'Not specified'.
            """
            
            llm = get_vision_llm()
            
            # Create message for LangChain ChatGroq Vision
            mime_type = f"image/{'jpeg' if file_extension == 'jpg' else file_extension}"
            msg = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
                ]
            )
            
            response = llm.invoke([msg])
            return response.content
            
        except Exception as e:
            return f"Error analyzing image: {e}"
    else:
        return f"Unsupported file type: {file_extension}. Please upload PNG, JPG, or PDF."
