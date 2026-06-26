import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

st.set_page_config(layout="wide", page_title="AI Data Auditor")

# 1. SIDEBAR CHATBOT (Anchored at the top so it never disappears)
st.sidebar.title("💬 Gemini Data Assistant")
if "messages" not in st.session_state: st.session_state.messages = []
for msg in st.session_state.messages:
    with st.sidebar.chat_message(msg["role"]): st.write(msg["content"])

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# 2. ROBUST DATA LOADER (Handles the string/mean error by force-converting)
def get_data(file):
    df = pd.read_csv(file)
    for col in df.columns:
        # Strip artifacts and force numeric if possible
        if df[col].dtype == 'object':
            clean = df[col].astype(str).str.replace(r'[,\$\s₹"\' ]', '', regex=True)
            if clean.str.replace('.', '', 1).str.isnumeric().mean() > 0.8:
                df[col] = pd.to_numeric(clean, errors='coerce')
    return df

# 3. MAIN DASHBOARD LOGIC
if uploaded_file:
    df = get_data(uploaded_file)
    st.title(f"Dashboard: {uploaded_file.name}")
    
    target = df.columns[-1]
    features = df.drop(columns=[target]).select_dtypes(include=[np.number]).columns
    
    tab1, tab2, tab3 = st.tabs(["📊 Analytics", "🤖 AI Models", "💡 Executive Summary"])
    
    with tab1:
        st.subheader("Descriptive Diagnostics")
        c1, c2 = st.columns(2)
        # Dynamic Bar Chart
        cat_cols = df.select_dtypes(exclude=[np.number]).columns
        if len(cat_cols) > 0:
            c1.plotly_chart(px.bar(df[cat_cols[0]].value_counts(), title=f"Distribution of {cat_cols[0]}"))
        # Numeric Histogram
        if len(features) > 0:
            c2.plotly_chart(px.histogram(df, x=features[0], color=target, title=f"Distribution of {features[0]}"))

    with tab2:
        st.subheader("Machine Learning Performance")
        if st.button("Train Models"):
            X = df[features].fillna(0)
            y = LabelEncoder().fit_transform(df[target].astype(str))
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "KNN": KNeighborsClassifier(),
                "Gradient Boosting": GradientBoostingClassifier()
            }
            
            results = []
            for name, model in models.items():
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                results.append({"Model": name, "Accuracy": accuracy_score(y_test, pred)})
            
            st.table(pd.DataFrame(results))
            st.success("All models trained!")

    with tab3:
        st.write("Dynamic insights based on your target column:", target)
        st.write(df.groupby(target).mean(numeric_only=True))

    # Gemini Integration
    if prompt := st.sidebar.chat_input("Ask Gemini about this data"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.sidebar.chat_message("user"): st.write(prompt)
        
        # Here you call your Gemini API logic
        response = "Gemini is analyzing your data..." 
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.sidebar.chat_message("assistant"): st.write(response)

else:
    st.info("Please upload a CSV to begin analysis.")
