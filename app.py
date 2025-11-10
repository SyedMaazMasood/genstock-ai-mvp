import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# --- Page Config ---
st.set_page_config(page_title="GenStock AI - CSV Analyst", layout="wide")
st.title("GenStock AI 📈 CSV Analyst")

# --- LLM Setup ---
# Use a fast and powerful model for the agent
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=groq_api_key)

st.info("Welcome! Upload your sales CSV file below to get started.")

# --- Session State Initialization ---
# Store the DataFrame, agent, and chat history
if "promo" not in st.session_state:
    st.session_state.promo = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "df" not in st.session_state:
    st.session_state.df = None
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Upload your data and ask me anything about it!"}]

# --- CSV Uploader ---
uploaded_file = st.file_uploader("Upload your sales data (CSV)", type="csv")

if uploaded_file:

    # Check if this is the first time uploading this file
    if st.session_state.agent_executor is None:
        st.session_state.df = df
        st.success("File uploaded successfully!")
        st.dataframe(df.head())
        
        with st.spinner("GenAI Agent is analyzing the data..."):
            st.session_state.agent_executor = create_pandas_dataframe_agent(
                llm,
                df,
                agent_type="openai-tools",
                verbose=True,
                allow_dangerous_code=True
            )
        
        # --- NEW: PROACTIVE SUMMARY ---
        with st.spinner("AI is generating your executive summary..."):
            try:
                summary_prompt = """
                Analyze the entire dataset. Provide a 3-bullet point executive summary for the store owner.
                Focus on:
                1. The best-selling item (by quantity) and its total revenue.
                2. The worst-selling item (by quantity).
                3. The total revenue across all items and the busiest day.
                """
                response = st.session_state.agent_executor.invoke({"input": summary_prompt})
                st.session_state.summary = response["output"]
            except Exception as e:
                st.error(f"Error generating summary: {e}")

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(uploaded_file)
    
    # Store the DataFrame in session state if it's not already there
    if st.session_state.df is None:
        st.session_state.df = df
        st.success("File uploaded successfully! You can now ask questions about your data.")
        
        # Display the first 5 rows of the data
        st.dataframe(df.head())
        
        # --- Create the Pandas Agent ---
        # We create the agent *once* and store it in session state
        with st.spinner("GenAI Agent is analyzing the data..."):
            st.session_state.agent_executor = create_pandas_dataframe_agent(
                llm,
                df,
                agent_type="openai-tools",
                verbose=True, # Set to True to see the agent's "thinking" in your terminal
                allow_dangerous_code=True
            )

# --- Main Interface ---
tab1, tab2 = st.tabs(["📈 Proactive Dashboard", "💬 Chat with Data"])

with tab2:
    # --- Display Chat History ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Chat Input ---
    if prompt := st.chat_input("Ask a follow-up question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Check if agent is loaded
        if st.session_state.agent_executor is None:
            st.error("Please upload a CSV file first.")
        else:
            # Generate response using the agent
            with st.chat_message("assistant"):
                with st.spinner("GenStock AI is thinking..."):
                    try:
                        response = st.session_state.agent_executor.invoke({
                            "input": prompt
                        })
                        response_content = response["output"]
                        st.markdown(response_content)
                        st.session_state.messages.append({"role": "assistant", "content": response_content})
                    except Exception as e:
                        st.error(f"An error occurred: {e}")

with tab1:
    st.header("Automated Business Insights")
    if st.session_state.df is None:
        st.info("Upload your CSV to see proactive insights from your GenStock AI.")
    else:
        st.markdown("Your AI agent has analyzed your sales data. Here's what it found:")
        # --- DISPLAY THE SUMMARY ---
        if st.session_state.summary:
            st.info(st.session_state.summary)
        
        st.markdown("---")
        st.subheader("🤖 GenAI Actions")
        
        if st.button("Suggest Promotion for Slowest Item", type="primary"):
            st.session_state.promo = "" # Clear old promo
            
            with st.spinner("AI is finding your slowest item..."):
                try:
                    # 1. Use the Pandas Agent to FIND the item
                    find_item_prompt = "What is the slowest-selling item (lowest total quantity sold)?"
                    item_response = st.session_state.agent_executor.invoke({"input": find_item_prompt})
                    slowest_item = item_response["output"]
                    
                    st.markdown(f"**Analysis:** The slowest item is: `{slowest_item}`")
                    
                    # 2. Define a "Promo Agent" (LLMChain) to GENERATE content
                    promo_template = """
                    You are a creative marketing assistant for a small convenience store.
                    Your goal is to reduce waste and sell slow-moving inventory.
                    
                    The store's slowest-moving item is: **{item}**
                    
                    Generate a short, catchy promotional idea to help sell this item.
                    Include:
                    1. A catchy promo name (e.g., "Daily Deal!", "BOGO Blast!").
                    2. A brief 1-2 sentence description for a sign.
                    3. A suggested discount (e.g., 25% off, Buy One Get One Free).
                    """
                    
                    promo_prompt = PromptTemplate(template=promo_template, input_variables=["item"])
                    promo_chain = LLMChain(llm=llm, prompt=promo_prompt)
                    
                    with st.spinner(f"AI is generating a promotion for {slowest_item}..."):
                        # 3. Run the Promo Agent
                        promo_response = promo_chain.invoke({"item": slowest_item})
                        st.session_state.promo = promo_response["text"]
                        
                except Exception as e:
                    st.error(f"Error generating promotion: {e}")

        # 4. Display the generated promotion
        if st.session_state.promo:
            st.markdown("---")
            st.header("Generated Promotion")
            st.success(st.session_state.promo)