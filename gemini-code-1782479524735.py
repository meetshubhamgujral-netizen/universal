"""
=========================================================================================
🔥 THE UNIVERSAL DATA-AGNOSTIC BIAS & ML SUITE WITH ANCHORED GEMINI ASSISTANT 🔥
=========================================================================================
Style Version: Streamlit Robust Production Layer (Zero-Crash Late-Binding Modules)
Features:
- Global Component Anchoring: Chatbot and layout sit at the top of the loop (always visible).
- Safe Datatype Coercion: Cleans currency punctuation formatting marks to eliminate mean errors.
- Context-Aware Native Analytics: Feeds exact data coordinates directly into conversational logic.
- Complete Super-Learning Suite: Runs Random Forest, Decision Tree, KNN, and Gradient Boosting.
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import streamlit as st

# --- CONTEXTUAL SAFE LATE-BINDING IMPORTS ---
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ModuleNotFoundError:
    HAS_PLOTLY = False

try:
    import google.generativeai as genai
    HAS_GEMINI_LIB = True
except ModuleNotFoundError:
    HAS_GEMINI_LIB = False

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
    HAS_SKLEARN = True
except ModuleNotFoundError:
    HAS_SKLEARN = False

# ---------------------------------------------------------------------------------------
# I. GLOBAL THEME DESIGN & BASE LIGHT LAYOUT INITIALIZATION
# ---------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Universal Business Intelligence & Super-Learning Suite",
    layout="wide",
    page_icon="⚖️",
    initial_sidebar_state="expanded"
)

# Enforce strict high-contrast font metrics globally to bypass theme text bleaching bugs
st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #F8FAFC !important;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        padding: 10px 20px !important;
        font-weight: bold !important;
    }
    .dynamic-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%);
        padding: 35px;
        border-radius: 12px;
        color: #FFFFFF !important;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.15);
    }
    .dynamic-header h1, .dynamic-header p { color: #FFFFFF !important; }
    .kpi-card {
        background-color: #FFFFFF !important;
        padding: 22px;
        border-radius: 12px;
        border-top: 6px solid #2563EB;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        text-align: center;
    }
    .kpi-card h2, .kpi-card p, .kpi-card span { color: #0F172A !important; }
    .dynamic-insight-card {
        background-color: #FFFFFF !important;
        border-left: 6px solid #10B981;
        padding: 20px;
        border-radius: 8px;
        margin-top: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .dynamic-insight-card h4, .dynamic-insight-card p, .dynamic-insight-card li, .dynamic-insight-card b { 
        color: #1E293B !important; 
    }
    .anomaly-alert-card {
        background-color: #FFFFFF !important;
        border-left: 6px solid #EF4444;
        padding: 22px;
        border-radius: 8px;
        margin-top: 15px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .anomaly-alert-card h4, .anomaly-alert-card p, .anomaly-alert-card li, .anomaly-alert-card b { 
        color: #7F1D1D !important; 
    }
    .empty-state-container {
        text-align: center;
        padding: 80px 20px;
        background-color: #FFFFFF;
        border-radius: 12px;
        border: 2px dashed #CBD5E1;
        margin-top: 40px;
    }
    h1, h2, h3, h4, h5, h6, p, label, span, div, small {
        color: #0F172A !important;
    }
    .stChatMessage p { color: #0F172A !important; }
    </style>
""", unsafe_allow_html=True)

# Preserve session chat data structures between panel events
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =======================================================================================
# II. GLOBAL ANCHORING PATTERN: USER FILE LOADER & ASSISTANT VIEW DRAWN AT ABSOLUTE START
# =======================================================================================
st.sidebar.markdown("### 📁 Universal Data Loader")
uploaded_file = st.sidebar.file_uploader("Upload Any Table Asset (CSV Format)", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 💬 Gemini Connected Data Mind")
chat_container = st.sidebar.container(height=360)
user_chat_input = st.sidebar.chat_input("Ask anything about your uploaded rows...")

with chat_container:
    if len(st.session_state.chat_history) == 0:
        st.markdown("<small style='color:#64748B;'>System active. Load a CSV spreadsheet to inject context models...</small>", unsafe_allow_html=True)
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# =======================================================================================
# III. BLANK WORKSPACE ONBOARDER (PREVENTS DUMMY OR BROKEN CRASH LOG DISPLAYS)
# =======================================================================================
if uploaded_file is None:
    st.markdown('<div class="dynamic-header"><h1>⚖️ Universal Adaptive Business Intelligence Suite</h1><p>Data-agnostic deep feature auditing suite running automated predictive transformations</p></div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="empty-state-container">
            <h3 style="color: #475569; margin-bottom: 10px;">📋 Awaiting Tabular File Entry Parameters</h3>
            <p style="color: #64748B; max-width: 500px; margin: 0 auto;">
                Please upload any data file into the system sidebar panel. The architecture will automatically map column definitions, isolate critical target vectors, build super-learning modeling configurations, and sync your live context-aware chatbot helper.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    if user_chat_input:
        st.sidebar.warning("💡 System requires baseline parameters. Upload a dataset file first to expose metric lookups.")
    st.stop()

# ---------------------------------------------------------------------------------------
# IV. DEFENSIVE DATA PROFILE MACHINE (BUILT FOR ANY GENERIC CSV STRUCTURE)
# ---------------------------------------------------------------------------------------
def auto_discover_schema(dataframe):
    """
    Dynamically identifies classification target columns, numeric properties, nominal properties,
    and row tracking unique identifiers without manual strings input.
    """
    schema = {
        'target_col': None,
        'age_col': None,
        'income_col': None,
        'group_col': None,
        'id_col': None,
        'numeric_features': [],
        'categorical_features': []
    }
    cols = dataframe.columns.tolist()
    
    # 1. Look for classification outcomes keywords dynamically
    for c in cols:
        c_upper = str(c).upper().strip()
        if any(kw in c_upper for kw in ['STATUS', 'ATTRITION', 'CHURN', 'TARGET', 'OUTPUT', 'DEFAULT', 'POLICY_STATUS', 'CLAIM_STATUS']):
            schema['target_col'] = c
            break
    if not schema['target_col']:
        schema['target_col'] = cols[-1]
        
    # 2. Look for explicit primary keys or record IDs
    for c in cols:
        c_upper = str(c).upper().strip()
        if any(kw in c_upper for kw in ['ID', 'POLICY_NO', 'EMPLOYEENUMBER', 'EMPLOYEE_NUMBER', 'SERIAL', 'PI_NAME', 'NAME']):
            schema['id_col'] = c
            break

    # 3. Classify data attributes strictly by parsed type structures
    for c in cols:
        if c == schema['target_col']: continue
        if pd.api.types.is_numeric_dtype(dataframe[c]):
            schema['numeric_features'].append(c)
            c_upper = str(c).upper().strip()
            if any(k in c_upper for k in ['AGE', 'BIRTH', 'PI_AGE', 'YEARS']):
                schema['age_col'] = c
            elif any(k in c_upper for k in ['INCOME', 'SALARY', 'EARNING', 'RATE', 'MONTHLYINCOME', 'DAILYRATE', 'PI_ANNUAL_INCOME', 'SUM_ASSURED']):
                schema['income_col'] = c
        else:
            schema['categorical_features'].append(c)
            c_upper = str(c).upper().strip()
            if any(k in c_upper for k in ['ZONE', 'DEPARTMENT', 'TEAM', 'BRANCH', 'LOCATION', 'STATE', 'JOBROLE', 'PI_STATE', 'GENDER']):
                schema['group_col'] = c
                
    if not schema['age_col'] and schema['numeric_features']: schema['age_col'] = schema['numeric_features'][0]
    if not schema['income_col'] and len(schema['numeric_features']) > 1: schema['income_col'] = schema['numeric_features'][1]
    if not schema['group_col'] and schema['categorical_features']: schema['group_col'] = schema['categorical_features'][0]
    
    return schema

def clean_tabular_types(raw_df):
    """
    Cleans structural tables by processing currency annotations and punctuation symbols,
    converting columns to true numeric floats before any reductions or mathematical summaries execute.
    """
    df_clean = raw_df.copy()
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            # Remove punctuation tokens and whitespace variables
            sample_stripped = df_clean[col].astype(str).str.replace(r'[,\s\$₹]', '', regex=True)
            if sample_stripped.str.replace('.', '', 1).str.isnumeric().sum() > 0.8 * len(raw_df) and len(raw_df) > 0:
                df_clean[col] = pd.to_numeric(sample_stripped, errors='coerce')
                
    # Fill structural nulls defensively using median values
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    for nc in num_cols:
        df_clean[nc] = df_clean[nc].fillna(df_clean[nc].median() if not df_clean[nc].isna().all() else 0)
        
    for col in df_clean.select_dtypes(include=['object', 'category']).columns:
        df_clean[col] = df_clean[col].fillna('UNKNOWN').astype(str).str.strip().str.upper()
        
    return df_clean

# Parse file metrics defensively
raw_data_frame = pd.read_csv(uploaded_file)
df = clean_tabular_types(raw_data_frame)
schema = auto_discover_schema(df)

target_var = schema['target_col']
age_var = schema['age_col']
income_var = schema['income_col']
group_var = schema['group_col']
id_var = schema['id_col']

unique_target_classes = df[target_var].unique().tolist()
risk_class_counts = df[target_var].value_counts()
risk_anchor_target = risk_class_counts.index[-1] if len(risk_class_counts) > 1 else unique_target_classes[0]
safe_anchor_target = risk_class_counts.index[0] if len(risk_class_counts) > 1 else unique_target_classes[0]

total_records = len(df)
total_risk_cases = len(df[df[target_var] == risk_anchor_target])
base_risk_rate = (total_risk_cases / total_records) * 100 if total_records > 0 else 0

# ---------------------------------------------------------------------------------------
# V. SYNCHRONIZING REAL-TIME CONTEXT DATA MATRIX TO THE GEMINI INTERFACE
# ---------------------------------------------------------------------------------------
if user_chat_input:
    st.session_state.chat_history.append({"role": "user", "content": user_chat_input})
    
    gemini_api_key = st.secrets.get("GEMINI_API_KEY", None)
    if HAS_GEMINI_LIB and gemini_api_key:
        try:
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel('gemini-pro')
            
            # Construct concrete dataset tracking boundary statements to eliminate hallucinations
            context_prompt = f"""
            You are a lead senior business intelligence and risk auditor data assistant.
            The user has uploaded a database spreadsheet containing exactly {total_records} evaluation data records.
            Structural metadata column metrics:
            - Goal Prediction Column Focus: '{target_var}' (Discovered status outcomes: {unique_target_classes})
            - Primary Risk Outcome Target Node: '{risk_anchor_target}' (Flagged instances: {total_risk_cases}, baseline incidence probability: {base_risk_rate:.2f}%)
            - Primary Stable Baseline Outcome State: '{safe_anchor_target}'
            - Continuous Metrics Variables: Age Key Axis='{age_var}', Financial scale matrix='{income_var}'
            - Primary Segment Operational Splitting Group: '{group_var}'
            - Unique Entity Key ID Tracker: '{id_var}'

            Active spreadsheet header view slice parameters:
            {df.head(5).to_string()}

            Answer all investigative queries comprehensively using this data snapshot context. If the user asks for specific high-risk records, names, or accounts who are highly susceptible to leaving or being repudiated, check rows where '{target_var}' equals '{risk_anchor_target}'. Keep responses clean, precise, and professional.
            """
            
            chat_chain_payload = f"{context_prompt}\n\nRecent Thread Chain logs:\n"
            for conversation_node in st.session_state.chat_history[-6:]:
                chat_chain_payload += f"{conversation_node['role'].upper()}: {conversation_node['content']}\n"
                
            gemini_response = model.generate_content(chat_chain_payload)
            st.session_state.chat_history.append({"role": "assistant", "content": gemini_response.text})
        except Exception as e:
            st.session_state.chat_history.append({"role": "assistant", "content": f"❌ Gemini API Interface encountered a lookup error: {str(e)}"})
    else:
        # Secure native calculations fallback matrix mapping if api keys are absent
        matched_target_queries = df[df[target_var] == risk_anchor_target].head(5)
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": f"📊 [Analytical Fallback Mode] Active records context processed. Portfolio features matching risk outcome state '{risk_anchor_target}' include {total_risk_cases} entries. Sample match lists tracker:\n\n{matched_target_queries[[id_var, group_var] if id_var and group_var else df.columns[:2].tolist()].to_string()}"
        })
        
    st.rerun()

# ---------------------------------------------------------------------------------------
# VI. VOCABULARY ENGINE SWITCH (DETERMINES CORE TERMINOLOGY VISUAL REWRITES)
# ---------------------------------------------------------------------------------------
clean_domain_name = str(target_var).replace('_', ' ').title()
if "ATTRITION" in str(target_var).upper():
    title_text = "Employee Attrition Risk & Staff Retention Analytics Suite"
    kpi_risk_title = f"ACTIVE ATTRITION STAFF COUNT ({risk_anchor_target})"
    kpi_safe_title = f"RETAINED STAFF ACTIVE COUNT ({safe_anchor_target})"
elif "STATUS" in str(target_var).upper() or "CLAIM" in str(target_var).upper():
    title_text = "Insurance Claim Settlement Bias & Adjudication Suite"
    kpi_risk_title = f"REPUDIATED PORTFOLIO CLAIMS ({risk_anchor_target})"
    kpi_safe_title = f"APPROVED PORTFOLIO CLAIMS ({safe_anchor_target})"
else:
    title_text = f"Enterprise {clean_domain_name} Optimization Suite"
    kpi_risk_title = f"RISK OUTCOME INSTANCES ({risk_anchor_target})"
    kpi_safe_title = f"STABLE STATE INSTANCES ({safe_anchor_target})"

st.markdown(f'<div class="dynamic-header"><h1>⚖️ {title_text}</h1><p>Data-agnostic analytical pipeline tracking active benchmarks for parameter: <b>{target_var}</b></p></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------------------
# VII. HIGH-CONTRAST DATA INFRASTRUCTURE KPI TILES
# ---------------------------------------------------------------------------------------
grid_c1, grid_c2, grid_c3, grid_c4 = st.columns(4)
with grid_c1:
    st.markdown(f'<div class="kpi-card"><p style="color:#64748B;font-size:12px;margin:0;font-weight:700;">TOTAL AUDITED RECORDS</p><h2 style="color:#0F172A;margin:5px 0;">{total_records:,}</h2></div>', unsafe_allow_html=True)
with grid_c2:
    st.markdown(f'<div class="kpi-card" style="border-top-color:#EF4444;"><p style="color:#64748B;font-size:12px;margin:0;font-weight:700;">{kpi_risk_title}</p><h2 style="color:#EF4444;margin:5px 0;">{total_risk_cases:,}</h2><span style="color:#EF4444;font-size:12px;font-weight:bold;">{base_risk_rate:.2f}% Event Probability</span></div>', unsafe_allow_html=True)
with grid_c3:
    st.markdown(f'<div class="kpi-card" style="border-top-color:#10B981;"><p style="color:#64748B;font-size:12px;margin:0;font-weight:700;">{kpi_safe_title}</p><h2 style="color:#10B981;margin:5px 0;">{total_records - total_risk_cases:,}</h2><span style="font-size:12px;color:#10B981;font-weight:bold;">{100-base_risk_rate:.2f}% Concentration Balance</span></div>', unsafe_allow_html=True)
with grid_c4:
    unique_operational_segments = df[group_var].nunique() if group_var else 1
    st.markdown(f'<div class="kpi-card" style="border-top-color:#F59E0B;"><p style="color:#64748B;font-size:12px;margin:0;font-weight:700;">DISTINCT SUB-OPERATIONAL NODES</p><h2 style="color:#F59E0B;margin:5px 0;">{unique_operational_segments} Categories</h2><span style="font-size:12px;color:#64748B;">Indexed classification via {group_var}</span></div>', unsafe_allow_html=True)

tab_descriptive, tab_diagnostic, tab_modeling, tab_findings = st.tabs([
    "📋 Descriptive Slices", "🔍 Disparity Diagnostics", "🤖 Super-Learning Classifiers", "💡 Dynamic Executive Summary"
])

# =======================================================================================
# TAB 1: DESCRIPTIVE SLICES
# =======================================================================================
with tab_descriptive:
    st.markdown(f"### 📋 Categorical Contingency Matrix vs `{target_var}`")
    valid_nominal_cols = schema['categorical_features'] if schema['categorical_features'] else df.columns.tolist()
    selected_slice = st.selectbox("Select Target nominal Dimension for Breakdown Slicing Analysis:", valid_nominal_cols)
    
    if selected_slice:
        count_ct = pd.crosstab(df[selected_slice], df[target_var])
        pct_ct = pd.crosstab(df[selected_slice], df[target_var], normalize='index') * 100
        
        split1, split2 = st.columns(2)
        with split1:
            st.markdown("#### Case Allocation Volume Profile")
            st.dataframe(count_ct, use_container_width=True)
        with split2:
            st.markdown("#### Proportional Percentage Allocations Slices Within Sub-Group (%)")
            st.dataframe(pct_ct.style.format("{:.2f}%"), use_container_width=True)
            
        if HAS_PLOTLY:
            try:
                fig_proportions = px.bar(df, x=selected_slice, color=target_var, barmode='group',
                                         color_discrete_sequence=px.colors.qualitative.Bold,
                                         title=f"Volumetric Sub-Category Value Split Distributions: {selected_slice}")
                fig_proportions.update_layout(xaxis={'categoryorder': 'total descending'}, template="plotly_white")
                st.plotly_chart(fig_proportions, use_container_width=True)
            except Exception:
                pass

# =======================================================================================
# TAB 2: SYSTEMIC DISPARITY DIAGNOSTICS (100% NATIVE ERROR SAFEGARDED)
# =======================================================================================
with tab_diagnostic:
    st.markdown("### 🔍 Systemic Group Performance Disparity Diagnostic Probes")
    probe_selection = st.radio("Isolate Targeted Evaluation Feature Layer Node:", ["Continuous Attribute Profile A", "Continuous Attribute Profile B", "Categorical Group Segments"], horizontal=True)
    
    try:
        if probe_selection == "Continuous Attribute Profile A" and age_var and HAS_PLOTLY:
            st.subheader(f"Auditing Disparity Profile Layout: `{age_var}`")
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                fig_box = px.box(df, x=target_var, y=age_var, color=target_var, points="outliers", color_discrete_sequence=['#0284C7', '#EF4444'])
                fig_box.update_layout(template="plotly_white")
                st.plotly_chart(fig_box, use_container_width=True)
            with col_a2:
                df['DYNAMIC_QUANTILE_BINS'] = pd.qcut(df[age_var], q=min(4, df[age_var].nunique()), duplicates='drop').astype(str)
                q_ct_tab = pd.crosstab(df['DYNAMIC_QUANTILE_BINS'], df[target_var], normalize='index') * 100
                fig_q_bar = px.bar(q_ct_tab.reset_index(), x='DYNAMIC_QUANTILE_BINS', y=q_ct_tab.columns.tolist(), barmode='stack', color_discrete_sequence=['#0284C7', '#EF4444'])
                fig_q_bar.update_layout(template="plotly_white")
                st.plotly_chart(fig_q_bar, use_container_width=True)
                
        elif probe_selection == "Continuous Attribute Profile B" and income_var and HAS_PLOTLY:
            st.subheader(f"Auditing Disparity Profile Layout: `{income_var}`")
            col_i1, col_i2 = st.columns(2)
            with col_i1:
                fig_inc_box = px.box(df, x=target_var, y=income_var, color=target_var, points="outliers", color_discrete_sequence=['#10B981', '#EF4444'])
                fig_inc_box.update_layout(template="plotly_white")
                st.plotly_chart(fig_inc_box, use_container_width=True)
            with col_i2:
                calc_df = df[[target_var, income_var]].copy().dropna()
                calc_df[income_var] = pd.to_numeric(calc_df[income_var], errors='coerce')
                mean_financials = calc_df.groupby(target_var)[income_var].mean().reset_index()
                
                fig_mean_inc = px.bar(mean_financials, x=target_var, y=income_var, color=target_var, color_discrete_sequence=['#10B981', '#EF4444'])
                fig_mean_inc.update_layout(template="plotly_white")
                st.plotly_chart(fig_mean_inc, use_container_width=True)
                
        elif probe_selection == "Categorical Group Segments" and group_var and HAS_PLOTLY:
            st.subheader(f"Auditing Disparity Profile Layout: `{group_var}`")
            group_ct_pct = pd.crosstab(df[group_var], df[target_var], normalize='index') * 100
            if risk_anchor_target in group_ct_pct.columns:
                group_ct_pct_sorted = group_ct_pct.sort_values(by=risk_anchor_target, ascending=False)
            else:
                group_ct_pct_sorted = group_ct_pct
            fig_grp = px.bar(group_ct_pct_sorted.reset_index(), x=group_var, y=group_ct_pct_sorted.columns.tolist(), barmode='stack', color_discrete_sequence=['#0284C7', '#EF4444'])
            fig_grp.update_layout(template="plotly_white")
            st.plotly_chart(fig_grp, use_container_width=True)
    except Exception:
        st.markdown("<div class='dynamic-insight-card'><h4>📋 Profile Disparity Calculations Unavailable</h4><p>Not enough valid numerical distributions exist to execute this matrix computation layout.</p></div>", unsafe_allow_html=True)

# =======================================================================================
# TAB 3: COMPLETE MACHINE LEARNING PIPELINE SUITE
# =======================================================================================
with tab_modeling:
    st.header("🤖 Parallel Super-Learning Classifiers Performance Matrix")
    
    if not HAS_SKLEARN:
        st.markdown("<p style='color:#64748B;'>Algorithmic modeling estimators require installation verification structures.</p>", unsafe_allow_html=True)
    elif len(df[target_var].unique()) < 2:
        st.markdown("<div class='anomaly-alert-card'><h4>❌ Supervised Classifier Execution Halted</h4><p>Target column does not possess enough distinct classes to train estimation metrics.</p></div>", unsafe_allow_html=True)
    else:
        try:
            ml_df = df.copy()
            if 'DYNAMIC_QUANTILE_BINS' in ml_df.columns: ml_df = ml_df.drop(columns=['DYNAMIC_QUANTILE_BINS'])
            
            le_tgt = LabelEncoder()
            ml_df[target_var] = le_tgt.fit_transform(ml_df[target_var].astype(str))
            
            X = ml_df.drop(columns=[target_var])
            y = ml_df[target_var]
            
            num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
            num_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
            cat_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
            
            transformer_block = ColumnTransformer(transformers=[('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
            X_train_proc = transformer_block.fit_transform(X_train)
            X_test_proc = transformer_block.transform(X_test)
            
            classifiers = {
                'Random Forest Classifier': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=8, class_weight='balanced'),
                'Decision Tree Classifier': DecisionTreeClassifier(random_state=42, max_depth=6, class_weight='balanced'),
                'KNN Classifier': KNeighborsClassifier(n_neighbors=5, weights='distance'),
                'Gradient Boosting Classifier': GradientBoostingClassifier(random_state=42, n_estimators=100, learning_rate=0.1, max_depth=4)
            }
            
            metrics_store = []
            target_risk_index = le_tgt.transform([str(risk_anchor_target).upper()])[0] if str(risk_anchor_target).upper() in le_tgt.classes_ else 1
            
            for name, clf in classifiers.items():
                clf.fit(X_train_proc, y_train)
                preds = clf.predict(X_test_proc)
                metrics_store.append({
                    'Model Configuration': name,
                    'Train Accuracy': accuracy_score(y_train, clf.predict(X_train_proc)),
                    'Test Accuracy': accuracy_score(y_test, preds),
                    'Precision': precision_score(y_test, preds, pos_label=target_risk_index, zero_division=0),
                    'Recall': recall_score(y_test, preds, pos_label=target_risk_index, zero_division=0),
                    'F1-Score': f1_score(y_test, preds, pos_label=target_risk_index, zero_division=0)
                })
                
            performance_matrix_df = pd.DataFrame(metrics_store)
            st.dataframe(performance_matrix_df.style.format({
                'Train Accuracy': "{:.2%}", 'Test Accuracy': "{:.2%}", 'Precision': "{:.2%}", 'Recall': "{:.2%}", 'F1-Score': "{:.2%}"
            }), use_container_width=True)
            
            if HAS_PLOTLY:
                fig_metrics = px.bar(performance_matrix_df.melt(id_vars='Model Configuration'), x='Model Configuration', y='value', color='variable', barmode='group', template='plotly_white')
                st.plotly_chart(fig_metrics, use_container_width=True)
        except Exception:
            st.markdown("<div class='dynamic-insight-card'><h4>📋 Predictive Evaluation Matrix Interrupted</h4><p>Not enough columns match numeric parameters to map mathematical optimization lines.</p></div>", unsafe_allow_html=True)

# =======================================================================================
# TAB 4: STRATEGIC INSIGHT ANALYSIS REWRITE (NO PLACEHOLDER STRINGS)
# =======================================================================================
with tab_findings:
    st.markdown("### 💡 Automated Strategic Briefing & Action Register")
    
    st.markdown('<div class="dynamic-insight-card">', unsafe_allow_html=True)
    st.markdown('<h4>📌 Discovered Operational Sub-Group Disparities</h4>', unsafe_allow_html=True)
    
    has_insights = False
    try:
        if age_var and age_var in df.columns:
            df['DYNAMIC_QUANTILE_BINS'] = pd.qcut(df[age_var], q=min(4, df[age_var].nunique()), duplicates='drop').astype(str)
            
        live_age_tab = pd.crosstab(df['DYNAMIC_QUANTILE_BINS'], df[target_var], normalize='index') * 100 if 'DYNAMIC_QUANTILE_BINS' in df.columns else pd.DataFrame()
        live_group_ct = pd.crosstab(df[group_var], df[target_var], normalize='index') * 100 if group_var else pd.DataFrame()
        
        if not live_group_ct.empty and risk_anchor_target in live_group_ct.columns:
            worst_grp = live_group_ct.sort_values(by=risk_anchor_target, ascending=False).index[0]
            worst_grp_val = live_group_ct.sort_values(by=risk_anchor_target, ascending=False)[risk_anchor_target].iloc[0]
            st.markdown(f"<li><b>Segment Disparity Filter Risk:</b> Case evaluation analysis indicates non-uniform distribution properties across organizational layers. The tracking node group identified as <b>{worst_grp}</b> holds the highest relative risk signature profile, displaying a <b>{worst_grp_val:.2f}% validation frequency for the {risk_anchor_target} risk track state</b>. This configuration highlights a key area for operational standardization.</li>", unsafe_allow_html=True)
            has_insights = True
            
        if income_var:
            calc_df = df[[target_var, income_var]].copy().dropna()
            calc_df[income_var] = pd.to_numeric(calc_df[income_var], errors='coerce')
            m_risk = calc_df[calc_df[target_var] == risk_anchor_target][income_var].mean()
            m_stable = calc_df[calc_df[target_var] == safe_anchor_target][income_var].mean()
            
            if not np.isnan(m_risk) and not np.isnan(m_stable):
                st.markdown(f"<li><b>Continuous Metric Concentration Filter:</b> Records associated with the <b>{risk_anchor_target}</b> risk path carry an average metrics profile value for `{income_var}` of <b>{m_risk:,.2f}</b>, compared to stable configurations averaging a profile of <b>{m_stable:,.2f}</b>.</li>", unsafe_allow_html=True)
                has_insights = True
    except Exception:
        pass
        
    if not has_insights:
        st.markdown("<p>Not enough unique distribution patterns exist to map programmatic summary insight summaries.</p>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
