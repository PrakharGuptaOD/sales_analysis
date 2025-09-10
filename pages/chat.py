import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import glob
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

st.set_page_config(layout="wide")
st.title("🤖 Chat with Multiple CSVs")

# 1. Check for the .env file and API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.warning("⚠️ **Gemini API Key not found.** Please set up your API key to continue.")
    st.info(
        """
        **How to set up your API Key:**
        1.  Create a new file named `.env` in the same directory as this script.
        2.  Inside the `.env` file, add the following line:
            `GEMINI_API_KEY="your_api_key_here"`
        3.  Replace `"your_api_key_here"` with your actual Gemini API key.
        4.  Save the file and rerun the app.
        
        You can get your key from the [Google AI Studio](https://aistudio.google.com/app/apikey).
        """
    )
    st.stop()

# 2. Configure the Generative AI library with the key from the .env file
try:
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"Failed to configure the API key. Please check if it is correct. Error: {e}")
    st.stop()

# ---
# Automated Data Loading from a 'datasets' Folder
# ---

# Define the folder path
DATA_FOLDER = "datasets"

# Check if the datasets folder exists
if not os.path.exists(DATA_FOLDER):
    st.error(f"The '{DATA_FOLDER}' folder was not found. Please create this folder and place your CSV files inside.")
    st.stop()

# Find and load all CSV files from the folder
try:
    csv_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
    
    if not csv_files:
        st.warning(f"No CSV files found in the '{DATA_FOLDER}' folder. Please add at least one CSV file.")
        st.stop()

    list_of_dfs = [pd.read_csv(file) for file in csv_files]
    df = pd.concat(list_of_dfs, ignore_index=True)

    st.success(f"Successfully loaded {len(csv_files)} file(s) from the '{DATA_FOLDER}' folder.")
    st.dataframe(df.head())

    prompt = st.text_input("Ask a question to generate analysis code (e.g., 'What is the average value in the Sales column?')")

    if prompt:
        model = genai.GenerativeModel('gemini-2.5-flash')
        column_names = ", ".join(df.columns)
        
        full_prompt = f"""
        You are an expert Python data analyst. You are given a pandas DataFrame named `df`.
        The columns in the DataFrame are: {column_names}.
        
        Your task is to write a short Python script to answer the following user question.
        - Only generate the Python code required to produce the answer.
        - Do not include any explanations, comments, or introductory text.
        - The result of the script should be stored in a variable called `result`.
        - Result should be in textual format suitable for display.

        
        User Question: {prompt}
        """

        with st.spinner("Generating analysis code..."):
            response = model.generate_content(full_prompt)
            code_to_execute = response.text.strip().replace("```python", "").replace("```", "")
        
        st.write("### Generated Code:")
        st.code(code_to_execute)

        with st.spinner("Executing code..."):
            local_vars = {"df": df}
            exec(code_to_execute, {"pd": pd}, local_vars)
            result = local_vars.get('result', "No result found in the generated code.")
            
            st.write("### Analysis Result:")
            st.write(result)

except Exception as e:
    st.error(f"An error occurred during analysis: {e}")
