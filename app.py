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

st.set_page_config(page_title="GenStock AI", layout="centered")
st.title("GenStock AI – Smart Inventory + Reordering")
st.caption("Zero-cost MVP using Llama 3.1 70B (Groq) + GPT-4o-mini")

# === MODELS (FREE!) ===
llama = ChatGroq(model="llama-3.1-70b-instant", temperature=0.7, api_key=os.getenv("GROQ_API_KEY"))
gpt_mini = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# === SIMPLE STOCK DETECTION FUNCTION (no tool decorator) ===
def detect_stock(image_bytes: bytes) -> str:
    """Detect items from shelf photo"""
    try:
        reader = easyocr.Reader(['en'], gpu=False)
        result = reader.readtext(image_bytes, detail=0)
        return ", ".join(result) if result else "No items detected"
    except:
        return "Croissants, RedBull 8-pack, 2% Milk Gallon"  # Mock fallback

# === AGENTS (no tools needed for demo) ===
inventory_agent = Agent(
    role="Inventory Specialist",
    goal="Accurately detect stock from shelf photos",
    backstory="You are an expert at reading messy store shelves",
    llm=llama,
    allow_delegation=False
)

reorder_agent = Agent(
    role="Smart Reordering Specialist",
    goal="Generate 3 perfect reorder drafts: WhatsApp, JSON, PDF",
    backstory="You know every vendor's format and never over-order",
    llm=llama,
    allow_delegation=False
)

# === SESSION STATE ===
if "stock" not in st.session_state:
    st.session_state.stock = {}
if "drafts" not in st.session_state:
    st.session_state.drafts = {}

# === SIDEBAR ===
with st.sidebar:
    st.header("Upload Shelf Photo")
    uploaded_file = st.file_uploader("Take or upload photo", type=["jpg", "png", "jpeg"])
    
    if uploaded_file and st.button("Analyze Stock"):
        with st.spinner("Detecting items..."):
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Shelf", use_column_width=True)
            img_bytes = uploaded_file.getvalue()
            result = detect_stock(img_bytes)
            st.success(f"Detected: {result}")
            
            # Mock stock for demo
            st.session_state.stock = {
                "Croissants": 4,
                "RedBull 8-pack": 6,
                "2% Milk Gallon": 12
            }

# === MAIN UI ===
if st.session_state.stock:
    st.write("### Current Stock")
    for item, qty in st.session_state.stock.items():
        color = "🟢" if qty > 10 else "🟡" if qty > 5 else "🔴"
        st.write(f"{color} **{item}**: {qty} units")

    if st.button("Run Smart Reorder", type="primary"):
        with st.spinner("Agents are thinking..."):
            task = Task(
                description=f"""
                Current stock: {st.session_state.stock}
                Par levels: Croissants=20, RedBull=24, Milk=30
                Vendor prefers: WhatsApp, web form, or in-person PDF.
                Generate 3 reorder drafts with short explanations.
                """,
                agent=reorder_agent,
                expected_output="WhatsApp text, JSON, PDF content"
            )
            crew = Crew(agents=[reorder_agent], tasks=[task], verbose=0)
            result = crew.kickoff()
            
            # Parse result (Llama is chatty)
            text = str(result)
            st.session_state.drafts = {
                "whatsapp": "Hey Mike, can we get 3 cases RedBull 8-pack for Thursday? We're at 6 units — usually sell 4/day. Total $198. Thanks! – Alex @ 7-Eleven #142",
                "json": '{"item": "RedBull 8-pack", "qty": 36, "date": "2025-11-21"}',
                "pdf": "RedBull 8-pack x 36 units\nDelivery: Nov 21\nTotal: $198.00"
            }
            if "WhatsApp" in text:
                st.session_state.drafts["whatsapp"] = text.split("WhatsApp")[1].split("\n")[0].strip()

# === APPROVAL QUEUE ===
if st.session_state.drafts:
    st.write("### Human Approval Gate")
    st.warning("Nothing leaves without your approval!")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**WhatsApp Text**")
        st.code(st.session_state.drafts["whatsapp"])
        if st.button("Approve & Copy", key="wa"):
            pyperclip.copy(st.session_state.drafts["whatsapp"])
            st.success("Copied to clipboard!")

    with col2:
        st.write("**Web Form JSON**")
        st.code(st.session_state.drafts["json"])
        if st.button("Approve & Copy", key="json"):
            pyperclip.copy(st.session_state.drafts["json"])
            st.success("JSON copied!")

    with col3:
        st.write("**PDF Order Sheet**")
        st.code(st.session_state.drafts["pdf"])
        if st.button("Download PDF", key="pdf"):
            buffer = io.BytesIO()
            c = canvas.Canvas(buffer, pagesize=letter)
            c.setFont("Helvetica-Bold", 16)
            c.drawString(100, 750, "GENSTOCK AI ORDER")
            c.setFont("Helvetica", 12)
            textobject = c.beginText(100, 720)
            for line in st.session_state.drafts["pdf"].split('\n'):
                textobject.textLine(line)
            c.drawText(textobject)
            c.save()
            buffer.seek(0)
            st.download_button("Download PDF", buffer, "order.pdf", "application/pdf")