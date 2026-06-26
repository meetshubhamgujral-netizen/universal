import streamlit as st
import pandas as pd
import numpy as np

# 1. ANCHORED CHATBOT & UPLOADER (These will NEVER disappear)
st.set_page_config(layout="wide")
st.sidebar.title("💬 Gemini Assistant")

if "messages" not in st.session_state: st.session_state.messages = []
for msg in st.session_state.messages:
    with st.sidebar.chat_message(msg["role"]): st.write(msg["content"])

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# 2. DEFENSIVE DATA LOADER
def process_data(file):
    df = pd.read_csv(file)
    # Automatically convert strings that look like numbers (Fixes the Mean/String Error)
    for col in df.columns:
        if df[col].dtype == 'object':
            clean = df[col].astype(str).str.replace(r'[,\$\s₹"\' ]', '', regex=True)
            if clean.str.replace('.', '', 1).str.isnumeric().mean() > 0.8:
                df[col] = pd.to_numeric(clean, errors='coerce')
    return df

# 3. DYNAMIC CONTENT
if uploaded_file:
    df = process_data(uploaded_file)
    st.title(f"Dashboard: {uploaded_file.name}")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Analytics", "🤖 AI Models", "💡 Executive Summary"])
    
    with tab1:
        st.subheader("Visual Analytics")
        target = df.columns[-1]
        try:
            import plotly.express as px
            st.plotly_chart(px.histogram(df, x=df.columns[0], color=target))
        except ImportError:
            st.bar_chart(df.select_dtypes(include=np.number).iloc[:, 0])

    with tab2:
        st.subheader("Machine Learning (Forest, Tree, KNN, Gradient)")
        if st.button("Run AI Models"):
            st.write("Training models on your data...")
            # Your ML logic here
            st.success("Analysis Complete!")

    # Gemini Chat logic
    if prompt := st.sidebar.chat_input("Ask Gemini..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Logic to call Gemini API
        st.sidebar.chat_message("assistant").write("Gemini is analyzing the data...")

else:
    st.info("Please upload a CSV file in the sidebar to begin.")
