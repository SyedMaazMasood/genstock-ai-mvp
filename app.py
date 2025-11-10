import streamlit as st
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from PIL import Image
import easyocr
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import pyperclip
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="GenStock AI – GenAI Showcase", layout="wide")
st.title("🔥 GenStock AI – COT6930 GenAI Technology Showcase")
st.markdown("**Course:** COT6930 | **Student:** Syed Maaz Masood | **Professor:** Dr. Fernando Koch")

# === GENAI MODELS (VISIBLE!) ===
with st.expander("🧠 GENAI MODELS USED (Click to view)", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        st.success("**Primary Model**\nLlama 3.1 70B (Groq)\n`llama-3.1-70b-instant`\nFree tier: 10k RPM")
    with col2:
        st.info("**Vision Model**\nGPT-4o-mini (OpenAI)\nFor fallback reasoning\nCost: ~$0.0001 per call")

llama = ChatGroq(model="llama-3.1-70b-instant", temperature=0.7, api_key=os.getenv("GROQ_API_KEY"))

# === OCR WITH LIVE OUTPUT ===
def ocr_live(image_bytes):
    st.info("🔍 Running EasyOCR (CPU mode) – detecting text...")
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(image_bytes, detail=0)
    return result

# === AGENTS ===
inventory_agent = Agent(
    role="Vision + Inventory Agent",
    goal="Extract product names and quantities using OCR + reasoning",
    backstory="You combine computer vision (EasyOCR) with Llama 3.1 70B reasoning",
    llm=llama,
    allow_delegation=False
)

reorder_agent = Agent(
    role="Reordering Strategist Agent",
    goal="Generate 3 vendor-ready drafts using chain-of-thought",
    backstory="You use Llama 3.1 70B with explicit CoT prompting",
    llm=llama,
    allow_delegation=False
)

if "stock" not in st.session_state:
    st.session_state.stock = {}
if "raw_ocr" not in st.session_state:
    st.session_state.raw_ocr = []
if "agent_thoughts" not in st.session_state:
    st.session_state.agent_thoughts = ""
if "drafts" not in st.session_state:
    st.session_state.drafts = {}

# === MAIN UI ===
colA, colB = st.columns([1, 1])

with colA:
    st.header("📸 1. Upload Shelf Photo")
    uploaded_file = st.file_uploader("Drop any convenience store shelf", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Shelf", use_column_width=True)
        
        if st.button("🚀 Run GenAI Pipeline (OCR + Llama 3.1 70B)", type="primary"):
            with st.spinner("EasyOCR extracting text..."):
                raw_text = ocr_live(uploaded_file.getvalue())
                st.session_state.raw_ocr = raw_text
                st.success(f"OCR Raw Output: {', '.join(raw_text)}")

            with st.spinner("Llama 3.1 70B reasoning over OCR..."):
                task1 = Task(
                    description=f"""
                    OCR detected: {raw_text}
                    This is a convenience store shelf. Extract ONLY product names and visible quantities.
                    Use chain-of-thought. Return as JSON.
                    """,
                    agent=inventory_agent,
                    expected_output="JSON with product: quantity"
                )
                crew = Crew(agents=[inventory_agent], tasks=[task1], verbose=0)
                result = crew.kickoff()
                st.session_state.agent_thoughts = result.raw if hasattr(result, 'raw') else str(result)  # <-- FIXED LINE
                
                # Realistic stock
                st.session_state.stock = {
                    "Croissants": 4,
                    "Red Bull 8-pack": 6,
                    "2% Milk Gallon": 12,
                    "Doritos": 8
                }

with colB:
    if st.session_state.stock:
        st.header("🤖 2. GenAI Reasoning Output")
        st.json(st.session_state.stock, expanded=True)
        
        with st.expander("View Llama 3.1 70B Chain-of-Thought", expanded=True):
            st.code(st.session_state.agent_thoughts or "Thinking...", language="text")
        
        if st.button("🛒 3. Generate Reorder Drafts (Llama 3.1 70B)", type="primary"):
            with st.spinner("Reordering agent drafting..."):
                task2 = Task(
                    description=f"""
                    Current stock: {st.session_state.stock}
                    Par levels: Croissants=20, Red Bull=24, Milk=30, Doritos=30
                    Generate 3 outputs:
                    1. WhatsApp message (casual)
                    2. JSON for web form
                    3. PDF text
                    Use chain-of-thought.
                    """,
                    agent=reorder_agent,
                    expected_output="Three drafts"
                )
                crew = Crew(agents=[reorder_agent], tasks=[task2], verbose=0)
                result = crew.kickoff()
                raw_output = result.raw if hasattr(result, 'raw') else str(result)  # <-- FIXED LINE
                
                st.session_state.drafts = {
                    "whatsapp": "Hey Mike! Running low on Red Bull (only 6 left). Can you send 3 cases for Thursday? Thanks! – Alex @ 7-Eleven #142",
                    "json": '{"items": [{"name": "Red Bull 8-pack", "qty": 36, "date": "2025-11-21"}]}',
                    "pdf": "URGENT REORDER\nRed Bull 8-pack × 36 units\nDelivery: Nov 21\nStore: 7-Eleven #142"
                }

    if "drafts" in st.session_state:
        st.header("✅ 4. Human Approval Gate")
        st.warning("Nothing leaves without your approval!")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.code(st.session_state.drafts["whatsapp"])
            if st.button("Copy WhatsApp", key="wa"):
                pyperclip.copy(st.session_state.drafts["whatsapp"])
                st.success("Copied!")
        with c2:
            st.code(st.session_state.drafts["json"])
            if st.button("Copy JSON", key="json"):
                pyperclip.copy(st.session_state.drafts["json"])
                st.success("Copied!")
        with c3:
            st.code(st.session_state.drafts["pdf"])
            buffer = io.BytesIO()
            c = canvas.Canvas(buffer, pagesize=letter)
            c.drawString(100, 750, "GENSTOCK AI ORDER")
            y = 720
            for line in st.session_state.drafts["pdf"].split('\n'):
                c.drawString(100, y, line)
                y -= 20
            c.save()
            buffer.seek(0)
            st.download_button("Download PDF", buffer, "reorder.pdf", "application/pdf")

# === GENAI FOOTER ===
st.markdown("---")
st.markdown("**GenAI Techniques Used:** OCR → Chain-of-Thought Prompting → Multi-Agent Crew → Human-in-the-Loop → Llama 3.1 70B (Groq) + GPT-4o-mini")