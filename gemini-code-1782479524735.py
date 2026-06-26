import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(layout="wide")

# 1. SIDEBAR: Persistent Chat
st.sidebar.title("💬 Gemini Data Assistant")
if "messages" not in st.session_state: st.session_state.messages = []
for msg in st.session_state.messages:
    with st.sidebar.chat_message(msg["role"]): st.write(msg["content"])

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# 2. DATA LOAD & CLEAN
def get_data(file):
    df = pd.read_csv(file)
    for col in df.columns:
        if df[col].dtype == 'object':
            clean = df[col].astype(str).str.replace(r'[,\$\s₹"\' ]', '', regex=True)
            if clean.str.replace('.', '', 1).str.isnumeric().mean() > 0.8:
                df[col] = pd.to_numeric(clean, errors='coerce')
    return df

# 3. MAIN APP
if uploaded_file:
    df = get_data(uploaded_file)
    st.title(f"Dashboard: {uploaded_file.name}")
    target = df.columns[-1]
    
    tab1, tab2, tab3 = st.tabs(["📊 Analytics", "🤖 AI Models", "💡 Executive Summary"])
    
    with tab1:
        st.subheader("Visual Analytics")
        # Explicitly drawing charts
        col_select = st.selectbox("Select numeric feature to view", df.select_dtypes(include=np.number).columns)
        st.plotly_chart(px.histogram(df, x=col_select, color=target, title=f"Distribution of {col_select}"))

    with tab2:
        st.subheader("Machine Learning Performance")
        if st.button("Run AI Models"):
            X = df.select_dtypes(include=[np.number]).fillna(0)
            y = LabelEncoder().fit_transform(df[target].astype(str))
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            
            models = {"Random Forest": RandomForestClassifier(), "Decision Tree": DecisionTreeClassifier(), "KNN": KNeighborsClassifier(), "Gradient Boosting": GradientBoostingClassifier()}
            results = []
            for name, m in models.items():
                m.fit(X_train, y_train)
                results.append({"Model": name, "Accuracy": round(m.score(X_test, y_test), 3)})
            st.table(pd.DataFrame(results))
            st.success("Analysis Complete!")

    with tab3:
        st.subheader("Executive Insights")
        st.write("Average metrics grouped by target:", df.groupby(target).mean(numeric_only=True))

    # Gemini Integration
    if prompt := st.sidebar.chat_input("Ask Gemini..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Simulate API response
        response = f"Analyzing your data... the mean of {target} is {df[target].value_counts().index[0]}."
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
else:
    st.info("Please upload a CSV to begin.")
