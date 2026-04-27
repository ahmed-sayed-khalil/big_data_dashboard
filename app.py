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
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import io

# Page Configuration
st.set_page_config(layout="wide", page_title="Advanced ML Studio", page_icon="🚀")
st.title("🚀 Advanced Analytics & ML Studio")

# --- Helper Functions ---
@st.cache_data
def get_preprocessor(df_pd, target_col):
    """Cached preprocessor builder - fixes column mismatch errors"""
    X = df_pd.drop(columns=[target_col])
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')), 
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), 
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # FIXED: sparse=False
        ]), cat_cols)
    ])
    return preprocessor, num_cols, cat_cols

def get_model_binary(model):
    buf = io.BytesIO()
    joblib.dump(model, buf)
    return buf.getvalue()

def export_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# Session State
if 'model_models' not in st.session_state:
    st.session_state.model_models = {
        "Random Forest": RandomForestRegressor(random_state=42),
        "Linear Regression": LinearRegression(),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Decision Tree": DecisionTreeRegressor(random_state=42)
    }

# --- Sidebar ---
uploaded_file = st.sidebar.file_uploader("📁 Upload Dataset (CSV/Parquet)", type=['csv', 'parquet'])

if uploaded_file:
    try:
        # FIXED: Robust file loading
        if 'csv' in uploaded_file.name.lower():
            df = pl.read_csv(uploaded_file, infer_schema_length=10000)
        else:
            df = pl.read_parquet(uploaded_file)
        df_pd = df.to_pandas()
        
        st.sidebar.success(f"✅ Loaded {len(df_pd):,} rows, {len(df_pd.columns)} columns")
        target_col = st.sidebar.selectbox("🎯 Target Variable:", df_pd.columns)
        
        # Cache preprocessor for all tabs
        preprocessor, num_cols, cat_cols = get_preprocessor(df_pd, target_col)
        st.sidebar.info(f"📊 {len(num_cols)} numeric, {len(cat_cols)} categorical columns detected")
        
    except Exception as e:
        st.error(f"❌ File loading failed: {str(e)}")
        st.stop()
    
    tabs = st.tabs(["📊 Data Profiling", "🔧 Pipeline", "🏆 Benchmarking", "🩺 Diagnostics", "📥 Export"])

    # TAB 1: EDA - UNCHANGED (working fine)
    with tabs[0]:
        st.subheader("Data Explorer")
        st.dataframe(df_pd.head(50), use_container_width=True)
        
        numeric_df = df_pd.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            st.write("### Correlation Matrix")
            fig_corr = px.imshow(numeric_df.corr(), text_auto=True, aspect="auto", 
                               color_continuous_scale='RdBu_r')
            st.plotly_chart(fig_corr, use_container_width=True)
            
            st.write("### Interactive Pairplot")
            cols = numeric_df.columns.tolist()
            cols_to_plot = st.multiselect("Select columns:", cols, 
                                        default=cols[:3] if len(cols) >= 3 else cols)
            if len(cols_to_plot) > 1:
                df_sample = df_pd[cols_to_plot].sample(n=min(len(df_pd), 500), random_state=42)
                st.plotly_chart(px.scatter_matrix(df_sample), use_container_width=True)

    # TAB 2: Pipeline - FIXED
    with tabs[1]:
        st.subheader("Preprocessing Pipeline")
        st.info("✅ Preprocessor auto-configured with current data columns")
        st.json({
            "Numeric columns": len(num_cols),
            "Categorical columns": len(cat_cols),
            "Transformations": "Imputation + Scaling (num), Imputation + OneHot (cat)"
        })
        
        if st.button("🔄 Refresh Preprocessor", type="secondary"):
            preprocessor, num_cols, cat_cols = get_preprocessor(df_pd, target_col)
            st.success("✅ Preprocessor refreshed!")

    # TAB 3: Benchmarking - FULLY FIXED
    with tabs[2]:
        if st.button("🚀 Run Model Benchmark", type="primary"):
            with st.spinner("Running 3-fold cross-validation..."):
                X_full = df_pd.drop(columns=[target_col])
                y_full = df_pd[target_col]
                
                results = {}
                for name, model in st.session_state.model_models.items():
                    pipe = Pipeline([('pre', preprocessor), ('reg', model)])
                    scores = cross_val_score(pipe, X_full, y_full, cv=3, scoring='r2')
                    results[name] = scores.mean()
                
                res_df = pd.DataFrame(list(results.items()), columns=['Model', 'R² Score'])
                res_df = res_df.sort_values('R² Score', ascending=False)
                
                col1, col2 = st.columns([3,1])
                with col1:
                    st.write("### Model Performance Comparison")
                    st.plotly_chart(
                        px.bar(res_df, x='R² Score', y='Model', orientation='h',
                             color='R² Score', color_continuous_scale='viridis')
                    )
                with col2:
                    st.table(res_df.style.format({'R² Score': '{:.3f}'}))
            
            st.success("✅ Benchmark complete!")

    # TAB 4: Diagnostics - FULLY FIXED  
    with tabs[3]:
        model_choice = st.selectbox("Select Model:", list(st.session_state.model_models.keys()))
        if st.button("🩺 Analyze Model", type="primary"):
            with st.spinner("Analyzing model performance..."):
                X_full = df_pd.drop(columns=[target_col])
                y_full = df_pd[target_col]
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X_full, y_full, test_size=0.2, random_state=42
                )
                
                model = st.session_state.model_models[model_choice]
                pipe = Pipeline([('pre', preprocessor), ('reg', model)])
                pipe.fit(X_train, y_train)
                preds = pipe.predict(X_test)
                
                # Metrics
                r2 = r2_score(y_test, preds)
                mae = mean_absolute_error(y_test, preds)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("R² Score", f"{r2:.3f}")
                col2.metric("MAE", f"{mae:.3f}")
                col3.metric("Test Size", f"{len(X_test):,}")
            
            c1, c2 = st.columns(2)
            with c1:
                fig1 = px.scatter(x=y_test, y=preds, 
                                labels={'x':'Actual', 'y':'Predicted'},
                                title=f"{model_choice}: Actual vs Predicted")
                fig1.add_shape(type="line", x0=y_test.min(), y0=y_test.min(),
                             x1=y_test.max(), y1=y_test.max(), line=dict(color="red"))
                st.plotly_chart(fig1, use_container_width=True)
            
            with c2:
                residuals = y_test - preds
                fig2 = px.scatter(x=preds, y=residuals,
                                labels={'x':'Predicted', 'y':'Residuals'},
                                title="Residual Plot")
                fig2.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig2, use_container_width=True)

    # TAB 5: Export - FIXED
    with tabs[4]:
        st.subheader("Model & Data Export")
        model_choice = st.selectbox("Select model to export:", list(st.session_state.model_models.keys()))
        
        if st.button("💾 Train & Prepare Model", type="primary"):
            with st.spinner("Training final model on full dataset..."):
                X_full = df_pd.drop(columns=[target_col])
                y_full = df_pd[target_col]
                
                model = st.session_state.model_models[model_choice]
                final_pipe = Pipeline([('pre', preprocessor), ('reg', model)])
                final_pipe.fit(X_full, y_full)
                
                st.session_state.final_model = final_pipe
                st.session_state.model_name = f"{model_choice}_model.pkl"
                st.success("✅ Model trained and ready for download!")
        
        if 'final_model' in st.session_state:
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="💾 Download Model (.pkl)",
                    data=get_model_binary(st.session_state.final_model),
                    file_name=st.session_state.model_name,
                    mime="application/octet-stream"
                )
            with col2:
                st.download_button(
                    label="📊 Export Clean Data (CSV)",
                    data=export_to_csv(df_pd),
                    file_name="clean_data.csv",
                    mime="text/csv"
                )

else:
    st.info("👆 Please upload a CSV or Parquet file to get started!")
    st.balloons()
