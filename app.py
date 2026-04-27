import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import io

# Page Configuration
st.set_page_config(layout="wide", page_title="Advanced ML Studio", page_icon="🚀")
st.title("🚀 Advanced Analytics & ML Studio")

# --- Helper Functions ---
def get_model_binary(model):
    buf = io.BytesIO()
    joblib.dump(model, buf)
    return buf.getvalue()

def export_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- Session State Initialization ---
if 'model_models' not in st.session_state:
    st.session_state.model_models = {
        "Random Forest": RandomForestRegressor(),
        "Linear Regression": LinearRegression(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "Decision Tree": DecisionTreeRegressor()
    }

# --- Sidebar & File Management ---
uploaded_file = st.sidebar.file_uploader("📁 Upload Dataset (CSV/Parquet)", type=['csv', 'parquet'])

# Reset logic to prevent column mismatch errors
if 'last_file' not in st.session_state: st.session_state.last_file = None
if uploaded_file != st.session_state.last_file:
    st.session_state.last_file = uploaded_file
    if 'preprocessor' in st.session_state: del st.session_state.preprocessor
    if 'ready_to_download' in st.session_state: del st.session_state.ready_to_download

if uploaded_file:
    df = pl.read_csv(uploaded_file) if 'csv' in uploaded_file.name else pl.read_parquet(uploaded_file)
    df_pd = df.to_pandas()
    target_col = st.sidebar.selectbox("🎯 Target Variable:", df_pd.columns)
    
    tabs = st.tabs(["📊 Data Profiling", "🔧 Pipeline", "🏆 Benchmarking", "🩺 Diagnostics", "📥 Export"])

    # TAB 1: EDA
    with tabs[0]:
        st.subheader("Data Explorer")
        st.dataframe(df_pd.head(50), width=1000)
        numeric_df = df_pd.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            st.write("### Correlation Matrix")
            st.plotly_chart(px.imshow(numeric_df.corr(), text_auto=True, color_continuous_scale='RdBu_r'), width=1000)

    # TAB 2: Pipeline
    with tabs[1]:
        if st.button("Initialize Preprocessor"):
            X = df_pd.drop(columns=[target_col])
            num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
            st.session_state.preprocessor = ColumnTransformer(transformers=[
                ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
                ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), 
                                  ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))]), cat_cols)
            ])
            st.success("✅ Preprocessor initialized.")

    # TAB 3: Benchmarking
    with tabs[2]:
        if st.button("Run Model Benchmark"):
            if 'preprocessor' in st.session_state:
                with st.status("🚀 Running 3-fold cross-validation...", expanded=True) as status:
                    X, y = df_pd.drop(columns=[target_col]), df_pd[target_col]
                    X_proc = st.session_state.preprocessor.fit_transform(X)
                    results = {name: cross_val_score(model, X_proc, y, cv=3).mean() for name, model in st.session_state.model_models.items()}
                    status.update(label="✅ Benchmarking complete!", state="complete", expanded=False)
                res_df = pd.DataFrame.from_dict(results, orient='index', columns=['R2 Score'])
                st.bar_chart(res_df)
                st.table(res_df.sort_values(by='R2 Score', ascending=False))
            else:
                st.warning("⚠️ Please initialize the preprocessor in the 'Pipeline' tab first.")

    # TAB 4: Diagnostics
    with tabs[3]:
        model_choice = st.selectbox("Select Model:", list(st.session_state.model_models.keys()))
        if st.button("Analyze Model"):
            if 'preprocessor' in st.session_state:
                with st.status("🩺 Analyzing model performance...", expanded=True) as status:
                    X, y = df_pd.drop(columns=[target_col]), df_pd[target_col]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    pipe = Pipeline([('pre', st.session_state.preprocessor), ('reg', st.session_state.model_models[model_choice])])
                    pipe.fit(X_train, y_train)
                    preds = pipe.predict(X_test)
                    status.update(label="✅ Analysis complete!", state="complete", expanded=False)
                c1, c2 = st.columns(2)
                c1.plotly_chart(px.scatter(x=y_test, y=preds, title="Actual vs Predicted"))
                c2.plotly_chart(px.scatter(x=preds, y=(y_test - preds), title="Residual Plot"))
            else:
                st.warning("⚠️ Please initialize the preprocessor in the 'Pipeline' tab first.")

    # TAB 5: Export
    with tabs[4]:
        if 'preprocessor' in st.session_state:
            best_model = st.selectbox("Select model to save:", list(st.session_state.model_models.keys()))
            if st.button("Prepare Model"):
                pipe = Pipeline([('pre', st.session_state.preprocessor), ('reg', st.session_state.model_models[best_model])])
                pipe.fit(df_pd.drop(columns=[target_col]), df_pd[target_col])
                st.session_state.ready_to_download = pipe
            if 'ready_to_download' in st.session_state:
                st.download_button("💾 Download Model (.pkl)", data=get_model_binary(st.session_state.ready_to_download), file_name="model.pkl")
        else:
            st.info("Initialize the preprocessor to enable exports.")
else:
    st.info("👆 Upload a file to begin.")
