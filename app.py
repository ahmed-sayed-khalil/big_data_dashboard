import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import io
import warnings
from sklearn.exceptions import FitFailedWarning

# Suppress sklearn CV warnings temporarily
warnings.filterwarnings("ignore", category=FitFailedWarning)

st.set_page_config(layout="wide", page_title="Advanced ML Studio", page_icon="🚀")
st.title("🚀 Advanced Analytics & ML Studio")

# --- ROBUST Data Cleaning ---
def clean_data_for_ml(df_pd, target_col):
    """Aggressive cleaning to prevent CV failures"""
    print(f"Cleaning data: {len(df_pd)} rows initially")
    
    # 1. Target cleaning FIRST
    df_clean = df_pd.dropna(subset=[target_col]).copy()
    df_clean[target_col] = pd.to_numeric(df_clean[target_col], errors='coerce')
    df_clean = df_clean.dropna(subset=[target_col])
    
    # Check target variance
    target_std = df_clean[target_col].std()
    if target_std < 1e-6:
        st.error(f"❌ Target '{target_col}' has zero variance (constant value). Cannot train regression models.")
        st.stop()
    
    print(f"After target cleaning: {len(df_clean)} rows, target std: {target_std:.3f}")
    
    # 2. Feature cleaning
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.drop(target_col, errors='ignore')
    
    # Drop features with >90% NaN or constant values
    for col in numeric_cols:
        na_pct = df_clean[col].isna().mean()
        std_val = df_clean[col].std()
        if na_pct > 0.9 or std_val < 1e-6:
            df_clean = df_clean.drop(columns=[col])
            print(f"Dropped {col}: {na_pct:.1%} NaN or constant")
    
    df_clean = df_clean.dropna(subset=numeric_cols.tolist())
    df_clean = df_clean.reset_index(drop=True)
    
    print(f"Final clean data: {len(df_clean)} rows")
    if len(df_clean) < 10:
        st.error("❌ Too few rows after cleaning (<10). Cannot perform cross-validation.")
        st.stop()
    
    return df_clean

# --- Helper Functions ---
@st.cache_data
def get_preprocessor(df_pd, target_col):
    X = df_pd.drop(columns=[target_col])
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Limit categorical explosion
    cat_cols = cat_cols[:10]  # Max 10 cat cols
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')), 
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), 
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=10))
        ]), cat_cols)
    ])
    return preprocessor, num_cols, cat_cols

def safe_cross_val(pipe, X, y, model_name):
    """Robust CV with error handling"""
    try:
        # Use fewer folds if small data
        n_samples = len(X)
        cv = min(3, n_samples // 5)  # At least 5 per fold
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        scores = cross_val_score(pipe, X, y, cv=kf, scoring='r2', 
                               error_score=np.nan, n_jobs=1)  # Serial to debug
        return scores.mean(), scores.std()
    except Exception as e:
        st.warning(f"❌ {model_name} CV failed: {str(e)[:100]}")
        return np.nan, np.nan

def get_model_binary(model):
    buf = io.BytesIO()
    joblib.dump(model, buf)
    return buf.getvalue()

def export_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# Session State
if 'model_models' not in st.session_state:
    st.session_state.model_models = {
        "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, random_state=42),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Linear Regression": LinearRegression()
    }

# --- Sidebar ---
uploaded_file = st.sidebar.file_uploader("📁 Upload Dataset (CSV/Parquet)", type=['csv', 'parquet'])

if uploaded_file:
    try:
        if 'csv' in uploaded_file.name.lower():
            df = pl.read_csv(uploaded_file, infer_schema_length=10000)
        else:
            df = pl.read_parquet(uploaded_file)
        df_pd_raw = df.to_pandas()
        
        st.sidebar.success(f"✅ Loaded {len(df_pd_raw):,} rows")
        
    except Exception as e:
        st.error(f"❌ File loading failed: {str(e)}")
        st.stop()
    
    # CRITICAL: Clean data FIRST
    target_col = st.sidebar.selectbox("🎯 Target Variable:", df_pd_raw.columns)
    df_pd = clean_data_for_ml(df_pd_raw, target_col)
    
    # Now safe to create preprocessor
    preprocessor, num_cols, cat_cols = get_preprocessor(df_pd, target_col)
    
    tabs = st.tabs(["📊 Data Profiling", "🔧 Pipeline", "🏆 Benchmarking", "🩺 Diagnostics", "📥 Export"])

    # TAB 1: EDA
    with tabs[0]:
        st.subheader("🔍 Cleaned Data Explorer")
        st.dataframe(df_pd.head(20), use_container_width=True)
        
        # Target distribution
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows after cleaning", f"{len(df_pd):,}")
            st.metric("Target std", f"{df_pd[target_col].std():.3f}")
        with col2:
            fig_hist = px.histogram(df_pd[target_col], nbins=30, title="Target Distribution")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        numeric_df = df_pd.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            st.write("### Correlation Matrix")
            corr = numeric_df.corr()
            fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
            st.plotly_chart(fig_corr, use_container_width=True)

    # TAB 2: Pipeline
    with tabs[1]:
        st.subheader("✅ Preprocessing Pipeline")
        st.json({
            "Dataset": f"{len(df_pd):,} rows after cleaning",
            "Target": target_col,
            "Numeric features": len(num_cols),
            "Categorical features": len(cat_cols),
            "Transformations": "Median impute + Scale (num), Missing + OneHot≤10cats (cat)"
        })

    # TAB 3: Benchmarking - FIXED WITH SAFE_CV
    with tabs[2]:
        if st.button("🚀 Run Model Benchmark", type="primary"):
            with st.spinner("Running robust cross-validation..."):
                X_full = df_pd.drop(columns=[target_col])
                y_full = df_pd[target_col]
                
                results = []
                for name, model in st.session_state.model_models.items():
                    pipe = Pipeline([('pre', preprocessor), ('reg', model)])
                    mean_score, std_score = safe_cross_val(pipe, X_full, y_full, name)
                    results.append({"Model": name, "R² Mean": mean_score, "R² Std": std_score})
                
                res_df = pd.DataFrame(results).sort_values('R² Mean', ascending=False)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("### Model Performance")
                    st.plotly_chart(
                        px.bar(res_df, x='R² Mean', y='Model', 
                             error_x='R² Std', orientation='h',
                             color='R² Mean', color_continuous_scale='viridis')
                    )
                with col2:
                    st.dataframe(res_df.round(3), use_container_width=True)
            
            best_model = res_df.iloc[0]['Model']
            st.success(f"🏆 Best: **{best_model}** (R²={res_df.iloc[0]['R² Mean']:.3f})")

    # TAB 4: Diagnostics - FIXED
    with tabs[3]:
        model_choice = st.selectbox("Select Model:", list(st.session_state.model_models.keys()))
        if st.button("🩺 Analyze Model", type="primary"):
            with st.spinner("Full train/test analysis..."):
                X_full = df_pd.drop(columns=[target_col])
                y_full = df_pd[target_col]
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X_full, y_full, test_size=0.2, random_state=42
                )
                
                model = st.session_state.model_models[model_choice]
                pipe = Pipeline([('pre', preprocessor), ('reg', model)])
                pipe.fit(X_train, y_train)
                preds = pipe.predict(X_test)
                
                r2 = r2_score(y_test, preds)
                mae = mean_absolute_error(y_test, preds)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("R² Score", f"{r2:.3f}")
                col2.metric("MAE", f"{mae:.3f}")
                col3.metric("Test Samples", len(X_test))
            
            c1, c2 = st.columns(2)
            with c1:
                fig1 = px.scatter(x=y_test, y=preds, 
                                labels={'x':'Actual', 'y':'Predicted'},
                                title=f"{model_choice}: Actual vs Predicted")
                fig1.add_shape(type="line", x0=min(y_test.min(), preds.min()), 
                             y0=min(y_test.min(), preds.min()),
                             x1=max(y_test.max(), preds.max()), 
                             y1=max(y_test.max(), preds.max()), 
                             line=dict(color="red", dash="dash"))
                st.plotly_chart(fig1, use_container_width=True)
            
            with c2:
                residuals = y_test - preds
                fig2 = px.scatter(x=preds, y=residuals,
                                labels={'x':'Predicted', 'y':'Residuals'},
                                title="Residuals")
                fig2.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig2, use_container_width=True)

    # TAB 5: Export
    with tabs[4]:
        model_choice = st.selectbox("Select model to export:", list(st.session_state.model_models.keys()))
        if st.button("💾 Train Final Model", type="primary"):
            with st.spinner("Training on full dataset..."):
                X_full = df_pd.drop(columns=[target_col])
                y_full = df_pd[target_col]
                model = st.session_state.model_models[model_choice]
                final_pipe = Pipeline([('pre', preprocessor), ('reg', model)])
                final_pipe.fit(X_full, y_full)
                st.session_state.final_model = final_pipe
                st.session_state.model_name = f"{model_choice.replace(' ', '_')}.pkl"
                st.success("✅ Ready for download!")
        
        if 'final_model' in st.session_state:
            col1, col2 = st.columns(2)
            col1.download_button(
                "💾 Model (.pkl)", 
                get_model_binary(st.session_state.final_model),
                st.session_state.model_name,
                "application/octet-stream"
            )
            col2.download_button(
                "📊 Clean Data (CSV)", 
                export_to_csv(df_pd),
                "clean_data.csv",
                "text/csv"
            )

else:
    st.info("👆 Upload CSV/Parquet to start → Works with housing, boston, california datasets!")
