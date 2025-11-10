import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
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
llm = ChatGroq(model="llama-3.1-70b-instant", temperature=0, api_key=groq_api_key)

st.info("Welcome! Upload your sales CSV file below to get started.")

# --- Session State Initialization ---
# Store the DataFrame, agent, and chat history
if "df" not in st.session_state:
    st.session_state.df = None
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Upload your data and ask me anything about it!"}]

# --- CSV Uploader ---
uploaded_file = st.file_uploader("Upload your sales data (CSV)", type="csv")

if uploaded_file:
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

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input ---
if prompt := st.chat_input("What is my top selling item?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check if data is loaded
    if st.session_state.agent_executor is None:
        st.error("Please upload a CSV file first.")
        st.session_state.messages.append({"role": "assistant", "content": "Please upload a CSV file first."})
    else:
        # Generate response using the agent
        with st.chat_message("assistant"):
            with st.spinner("GenStock AI is thinking..."):
                try:
                    # Use the agent to get the response
                    response = st.session_state.agent_executor.invoke({
                        "input": prompt
                    })
                    
                    response_content = response["output"]
                    st.markdown(response_content)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response_content})
                
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": "I'm sorry, I ran into an error trying to answer that."})