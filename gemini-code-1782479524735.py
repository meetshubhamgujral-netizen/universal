"""
=========================================================================================
🔥 ENHANCED FLUID BUSINESS INTELLIGENCE SUITE WITH CONTEXT-AWARE GEMINI ASSISTANT 🔥
=========================================================================================
Style Version: Streamlit Robust Native Production Layer (Global UI Structural Stability)
Features:
- Global Component Anchoring: The chat engine resides at the top of the runtime loop.
- Late-Binding Safe Imports: Bypasses container startup lockouts entirely.
- Dynamic Blank State Onboarder: Shows clean messaging when no file is present.
- Pure Plotly Layout Architecture: No Matplotlib dependencies to prevent rendering crashes.
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------------------
# I. APPLICATION SUITE DESIGN CONFIGURATIONS & CRITICAL CONTRAST LAYERS
# ---------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Universal Business Intelligence Suite",
    layout="wide",
    page_icon="⚖️",
    initial_sidebar_state="expanded"
)

# Custom high-contrast light theme style variables preventing white text bleaching bugs
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

# Preserve session chat log statements consistently across dynamic switches
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =======================================================================================
# II. STEP 1 OF PERMANENT ANCHORING: DATA LOADER & ASSISTANT VIEW INITIALIZED AT TOP
# =======================================================================================
st.sidebar.markdown("### 📁 Universal Data Loader")
uploaded_file = st.sidebar.file_uploader("Upload Core Business Database (CSV Format)", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 💬 Gemini Live Data Assistant")
chat_container = st.sidebar.container(height=360)
user_chat_input = st.sidebar.chat_input("Ask about your dataset coordinates:")

with chat_container:
    if len(st.session_state.chat_history) == 0:
        st.markdown("<small style='color:#64748B;'>Awaiting document registry inputs to provide conversation memory context...</small>", unsafe_allow_html=True)
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# =======================================================================================
# III. BLANK STATE PRESENTATION LAYER (ELIMINATES ACCIDENTAL PLACEHOLDER COMPONENT RENDERS)
# =======================================================================================
if uploaded_file is None:
    st.markdown('<div class="dynamic-header"><h1>⚖️ Universal Corporate BI Analytics & Super-Learning Suite</h1><p>Data-agnostic analytical pipeline with integrated context-aware assistance capabilities</p></div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="empty-state-container">
            <h3 style="color: #475569; margin-bottom: 10px;">📋 Please upload a dataset to begin analysis</h3>
            <p style="color: #64748B; max-width: 500px; margin: 0 auto;">
                Use the sidebar upload panel to load any tabular data file (e.g., Insurance or Employee Attrition). The dashboard will automatically profile columns, calculate performance slices, and activate context-aware chatbot models without errors.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    if user_chat_input:
        st.sidebar.warning("💡 Awaiting file registry entry. Upload your database CSV ledger first to supply context loops.")
    st.stop()

# ---------------------------------------------------------------------------------------
# IV. ROBUST AUTOMATED SCHEMA DETECTOR & EXTRACTION UTILITY
# ---------------------------------------------------------------------------------------
def auto_discover_schema(dataframe):
    """
    Dynamically identifies the target classification vector, continuous tracking coordinates,
    nominal categories, and record identifier strings safely.
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
    
    for c in cols:
        c_upper = str(c).upper().strip()
        if any(kw in c_upper for kw in ['STATUS', 'ATTRITION', 'CHURN', 'TARGET', 'OUTPUT', 'DEFAULT', 'POLICY_STATUS']):
            schema['target_col'] = c
            break
    if not schema['target_col']:
        schema['target_col'] = cols[-1]
        
    for c in cols:
        c_upper = str(c).upper().strip()
        if any(kw in c_upper for kw in ['ID', 'POLICY_NO', 'EMPLOYEENUMBER', 'EMPLOYEE_NUMBER', 'SERIAL', 'PI_NAME']):
            schema['id_col'] = c
            break

    for c in cols:
        if c == schema['target_col']: continue
        if pd.api.types.is_numeric_dtype(dataframe[c]):
            schema['numeric_features'].append(c)
            c_upper = str(c).upper().strip()
            if any(k in c_upper for k in ['AGE', 'BIRTH', 'PI_AGE']):
                schema['age_col'] = c
            elif any(k in c_upper for k in ['INCOME', 'SALARY', 'EARNING', 'RATE', 'MONTHLYINCOME', 'DAILYRATE', 'PI_ANNUAL_INCOME', 'SUM_ASSURED']):
                schema['income_col'] = c
        else:
            schema['categorical_features'].append(c)
            c_upper = str(c).upper().strip()
            if any(k in c_upper for k in ['ZONE', 'DEPARTMENT', 'TEAM', 'BRANCH', 'LOCATION', 'STATE', 'JOBROLE', 'PI_STATE']):
                schema['group_col'] = c
                
    if not schema['age_col'] and schema['numeric_features']: schema['age_col'] = schema['numeric_features'][0]
    if not schema['income_col'] and len(schema['numeric_features']) > 1: schema['income_col'] = schema['numeric_features'][1]
    if not schema['group_col'] and schema['categorical_features']: schema['group_col'] = schema['categorical_features'][0]
    
    return schema

def clean_tabular_types(raw_df):
    """
    Cleans structural column vector groups by dropping character symbols and formatting metadata,
    casting targets into clear float numeric metrics to prevent mean calculation reduction crashes.
    """
    df_clean = raw_df.copy()
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            sample_stripped = df_clean[col].astype(str).str.replace(r'[,\s\$₹]', '', regex=True)
            if sample_stripped.str.replace('.', '', 1).str.isnumeric().sum() > 0.8 * len(raw_df) and len(raw_df) > 0:
                df_clean[col] = pd.to_numeric(sample_stripped, errors='coerce')
                
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    for nc in num_cols:
        df_clean[nc] = df_clean[nc].fillna(df_clean[nc].median() if not df_clean[nc].isna().all() else 0)
        
    for col in df_clean.select_dtypes(include=['object', 'category']).columns:
        df_clean[col] = df_clean[col].fillna('UNKNOWN').astype(str).str.strip().str.upper()
        
    return df_clean

# Parse loaded data tables defensively
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

# =======================================================================================
# V. LATE-BINDING SAFELY PROTECTED CONTEXT-AWARE GEMINI INTELLIGENCE ASSISTANT PIPELINE
# =======================================================================================
if user_chat_input:
    st.session_state.chat_history.append({"role": "user", "content": user_chat_input})
    
    try:
        import google.generativeai as genai
        gemini_api_key = st.secrets.get("GEMINI_API_KEY", None)
        
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel('gemini-pro')
            
            # Anchor precise data facts to provide baseline model alignment
            context_prompt = f"""
            You are a senior operational enterprise business intelligence risk analyst.
            The user has uploaded a file tracker dataset containing exactly {total_records} rows.
            Blueprint metadata fields context profiles:
            - Mapped Prediction Goal Column: '{target_var}' (Discovered outcomes: {unique_target_classes})
            - Elevated Critical Risk Target Node: '{risk_anchor_target}' (Occurrences: {total_risk_cases}, baseline incidence: {base_risk_rate:.2f}%)
            - Stable/Normal Outcome Target Node: '{safe_anchor_target}'
            - Continuous Evaluation Numeric Attributes: Age scale axis='{age_var}', Financial metric dimension='{income_var}'
            - Primary Segment Operational Splitting Category Column: '{group_var}'
            - Primary Unique Record Index Tracking Field Key: '{id_var}'

            Active data block header view sample matrix for query mappings:
            {df.head(5).to_string()}

            Answer all structural inquiries concisely using this dataset matrix snapshot. If the user asks for specific high-risk records, entities, or keys who are highly susceptible to leaving or being repudiated, check directly against data rows matching '{target_var}' equals '{risk_anchor_target}'. Maintain strict dark-text layout rendering.
            """
            
            chat_payload_string = f"{context_prompt}\n\nRecent Active Thread Loops:\n"
            for conversation_node in st.session_state.chat_history[-6:]:
                chat_payload_string += f"{conversation_node['role'].upper()}: {conversation_node['content']}\n"
                
            gemini_response = model.generate_content(chat_payload_string)
            st.session_state.chat_history.append({"role": "assistant", "content": gemini_response.text})
        else:
            st.session_state.chat_history.append({"role": "assistant", "content": "⚠️ Config token missing. Provide your `GEMINI_API_KEY` inside your cloud secrets workspace dashboard."})
    except ImportError:
        # Fallback to standard Python calculation framework context matching if library fails to load
        matched_target_queries = df[df[target_var] == risk_anchor_target].head(5)
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": f"📊 [Native Fallback Mode] I've analyzed your input parameters. Active risk instances counting for class '{risk_anchor_target}' include {total_risk_cases} instances. Sample risk IDs index matches:\n\n{matched_target_queries[[id_var, group_var] if id_var and group_var else df.columns[:2].tolist()].to_string()}"
        })
    except Exception as e:
        st.session_state.chat_history.append({"role": "assistant", "content": f"❌ Operational model connection issue encountered: {str(e)}"})
        
    st.rerun()

# ---------------------------------------------------------------------------------------
# VI. MAIN DOMAIN TITLES ADJUSTMENT ENGINE
# ---------------------------------------------------------------------------------------
clean_domain_name = str(target_var).replace('_', ' ').title()
if "ATTRITION" in str(target_var).upper():
    title_text = "Employee Attrition Risk & Retention Analytics Suite"
    kpi_risk_title = f"ACTIVE ATTRITION COUNT ({risk_anchor_target})"
    kpi_safe_title = f"RETAINED STAFF COUNT ({safe_anchor_target})"
elif "STATUS" in str(target_var).upper() or "CLAIM" in str(target_var).upper():
    title_text = "Insurance Claim Settlement Bias & Adjudication Suite"
    kpi_risk_title = f"REPUDIATED CLAIMS COUNT ({risk_anchor_target})"
    kpi_safe_title = f"APPROVED CLAIMS COUNT ({safe_anchor_target})"
else:
    title_text = f"Enterprise {clean_domain_name} Optimization Suite"
    kpi_risk_title = f"RISK STATE INSTANCES ({risk_anchor_target})"
    kpi_safe_title = f"STABLE STATE INSTANCES ({safe_anchor_target})"

st.markdown(f'<div class="dynamic-header"><h1>⚖️ {title_text}</h1><p>Data-agnostic analytical processing suite monitoring active parameters profile for: <b>{target_var}</b></p></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------------------
# VII. HIGH-CONTRAST DYNAMIC KPI CARDS STRIP
# ---------------------------------------------------------------------------------------
grid_c1, grid_c2, grid_c3, grid_c4 = st.columns(4)
with grid_c1:
    st.markdown(f'<div class="kpi-card"><p style="color:#64748B;font-size:12px;margin:0;font-weight:700;">TOTAL AUDITED PORFOLIOS</p><h2 style="color:#0F172A;margin:5px 0;">{total_records:,}</h2></div>', unsafe_allow_html=True)
with grid_c2:
    st.markdown(f'<div class="kpi-card" style="border-top-color:#EF4444;"><p style="color:#64748B;font-size:12px;margin:0;font-weight:700;">{kpi_risk_title}</p><h2 style="color:#EF4444;margin:5px 0;">{total_risk_cases:,}</h2><span style="color:#EF4444;font-size:12px;font-weight:bold;">{base_risk_rate:.2f}% Baseline Frequency Rate</span></div>', unsafe_allow_html=True)
with grid_c3:
    st.markdown(f'<div class="kpi-card" style="border-top-color:#10B981;"><p style="color:#64748B;font-size:12px;margin:0;font-weight:700;">{kpi_safe_title}</p><h2 style="color:#10B981;margin:5px 0;">{total_records - total_risk_cases:,}</h2><span style="font-size:12px;color:#10B981;font-weight:bold;">{100-base_risk_rate:.2f}% Concentration Balance</span></div>', unsafe_allow_html=True)
with grid_c4:
    unique_operational_segments = df[group_var].nunique() if group_var else 1
    st.markdown(f'<div class="kpi-card" style="border-top-color:#F59E0B;"><p style="color:#64748B;font-size:12px;margin:0;font-weight:700;">DISTINCT LOGICAL SEGMENTS</p><h2 style="color:#F59E0B;margin:5px 0;">{unique_operational_segments} Groups</h2><span style="font-size:12px;color:#64748B;">Indexed classification via {group_var}</span></div>', unsafe_allow_html=True)

# Initialize functional tabs
tab_descriptive, tab_diagnostic, tab_modeling, tab_findings = st.tabs([
    "📋 Descriptive Slices", "🔍 Disparity Diagnostics", "🤖 Super-Learning Classifiers", "💡 Dynamic Executive Summary"
])

# =======================================================================================
# TAB 1: DESCRIPTIVE ANALYSIS
# =======================================================================================
with tab_descriptive:
    st.header(f"📋 Cross-Tabulation Breakdown vs {target_var}")
    valid_nominal_cols = schema['categorical_features'] if schema['categorical_features'] else df.columns.tolist()
    selected_slice = st.selectbox("Select Target Variable for Distribution Slicing:", valid_nominal_cols)
    
    if selected_slice:
        count_ct = pd.crosstab(df[selected_slice], df[target_var])
        pct_ct = pd.crosstab(df[selected_slice], df[target_var], normalize='index') * 100
        
        split1, split2 = st.columns(2)
        with split1:
            st.markdown("#### Case Allocation Volume Profile")
            st.dataframe(count_ct, use_container_width=True)
        with split2:
            st.markdown("#### Proportional Allocation Slices Within Sub-Group (%)")
            st.dataframe(pct_ct.style.format("{:.2f}%"), use_container_width=True)
            
        try:
            import plotly.express as px
            fig_proportions = px.bar(df, x=selected_slice, color=target_var, barmode='group',
                                     color_discrete_sequence=px.colors.qualitative.Bold,
                                     title=f"Volumetric Sub-Category Metric Distribution Splitting: {selected_slice}")
            fig_proportions.update_layout(xaxis={'categoryorder': 'total descending'}, template="plotly_white")
            st.plotly_chart(fig_proportions, use_container_width=True)
        except Exception:
            st.markdown("<p style='color:#64748B;'>Not enough categorical feature distribution variants to draw visual bar groupings.</p>", unsafe_allow_html=True)

# =======================================================================================
# TAB 2: DISPARITY DIAGNOSTICS
# =======================================================================================
with tab_diagnostic:
    st.header("🔍 Dynamic Auditing Disparity Probe Engine")
    
    try:
        import plotly.express as px
        probe_selection = st.radio("Isolate Targeted Evaluation Feature Layer Node:", ["Continuous Attribute Profile A", "Continuous Attribute Profile B", "Categorical Group Segments"], horizontal=True)
        
        if probe_selection == "Continuous Attribute Profile A" and age_var:
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
                
        elif probe_selection == "Continuous Attribute Profile B" and income_var:
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
                
        elif probe_selection == "Categorical Group Segments" and group_var:
            group_ct_pct = pd.crosstab(df[group_var], df[target_var], normalize='index') * 100
            if risk_anchor_target in group_ct_pct.columns:
                group_ct_pct_sorted = group_ct_pct.sort_values(by=risk_anchor_target, ascending=False)
            else:
                group_ct_pct_sorted = group_ct_pct
            fig_grp = px.bar(group_ct_pct_sorted.reset_index(), x=group_var, y=group_ct_pct_sorted.columns.tolist(), barmode='stack', color_discrete_sequence=['#0284C7', '#EF4444'])
            fig_grp.update_layout(template="plotly_white")
            st.plotly_chart(fig_grp, use_container_width=True)
    except Exception:
        st.markdown("<div class='dynamic-insight-card'><h4>📋 Profile Disparity Map Unavialable</h4><p>Not enough data to generate this analysis.</p></div>", unsafe_allow_html=True)

# =======================================================================================
# TAB 3: MACHINE LEARNING BENCHMARK MATRIX
# =======================================================================================
with tab_modeling:
    st.header("🤖 Parallel Super-Learning Classifiers Performance Matrix")
    
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        if len(df[target_var].unique()) < 2:
            st.markdown("<p style='color:#64748B;'>Target column does not possess enough distinct classes to train estimation metrics.</p>", unsafe_allow_html=True)
        else:
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
                'KNN Classifier': KNeighborsClassifier(n_neighbors=5, weights='distance'),
                'Decision Tree Classifier': DecisionTreeClassifier(random_state=42, max_depth=6, class_weight='balanced'),
                'Random Forest Classifier': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=8, class_weight='balanced'),
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
            
            try:
                import plotly.express as px
                fig_metrics = px.bar(performance_matrix_df.melt(id_vars='Model Configuration'), x='Model Configuration', y='value', color='variable', barmode='group', template='plotly_white')
                st.plotly_chart(fig_metrics, use_container_width=True)
            except Exception:
                pass
    except Exception:
        st.markdown("<div class='dynamic-insight-card'><h4>📋 Supervised Learning Evaluation Unavialable</h4><p>Not enough data to generate this analysis.</p></div>", unsafe_allow_html=True)

# =======================================================================================
# TAB 4: AUTOMATED STRATEGIC DISPARITY BRIEFINGS
# =======================================================================================
with tab_findings:
    st.markdown("### 💡 Automated Strategic Briefing & Action Register")
    
    st.markdown('<div class="dynamic-insight-card">', unsafe_allow_html=True)
    st.markdown('<h4>📌 Identified Operational Risk Divergences</h4>', unsafe_allow_html=True)
    
    has_insights = False
    try:
        if age_var and age_var in df.columns:
            df['DYNAMIC_QUANTILE_BINS'] = pd.qcut(df[age_var], q=min(4, df[age_var].nunique()), duplicates='drop').astype(str)
            
        live_age_tab = pd.crosstab(df['DYNAMIC_QUANTILE_BINS'], df[target_var], normalize='index') * 100 if 'DYNAMIC_QUANTILE_BINS' in df.columns else pd.DataFrame()
        live_group_ct = pd.crosstab(df[group_var], df[target_var], normalize='index') * 100 if group_var else pd.DataFrame()
        
        if not live_group_ct.empty and risk_anchor_target in live_group_ct.columns:
            worst_grp = live_group_ct.sort_values(by=risk_anchor_target, ascending=False).index[0]
            worst_grp_val = live_group_ct.sort_values(by=risk_anchor_target, ascending=False)[risk_anchor_target].iloc[0]
            st.markdown(f"<li><b>Segment Disparity Variance Alert:</b> Portfolio distributions indicate non-uniform behavior traits across structural layers. The tracking node group matching <b>{worst_grp}</b> holds the highest relative risk signature profile, displaying a <b>{worst_grp_val:.2f}% validation frequency for the {risk_anchor_target} risk track state</b>. This configuration highlights a key area for operational standardization.</li>", unsafe_allow_html=True)
            has_insights = True
            
        if income_var:
            calc_df = df[[target_var, income_var]].copy().dropna()
            calc_df[income_var] = pd.to_numeric(calc_df[income_var], errors='coerce')
            m_risk = calc_df[calc_df[target_var] == risk_anchor_target][income_var].mean()
            m_stable = calc_df[calc_df[target_var] == safe_anchor_target][income_var].mean()
            
            if not np.isnan(m_risk) and not np.isnan(m_stable):
                st.markdown(f"<li><b>Continuous Metric Factor Scale Imbalance:</b> Profiles trending on critical **{risk_anchor_target}** paths carry an average metrics profile value for `{income_var}` of <b>{m_risk:,.2f}</b>, compared to stable configurations averaging a profile of <b>{m_stable:,.2f}</b>.</li>", unsafe_allow_html=True)
                has_insights = True
    except Exception:
        pass
        
    if not has_insights:
        st.markdown("<p>Not enough data to generate this analysis.</p>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)