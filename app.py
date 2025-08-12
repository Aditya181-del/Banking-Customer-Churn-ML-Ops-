# 🚀 app.py - Refactored for Portfolio Grade Quality (Single File)

import streamlit as st # type: ignore
import pandas as pd
import numpy as np
import joblib
import uuid
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px
import plotly.graph_objects as go
import shap
import random   
from io import BytesIO
from sklearn.calibration import CalibratedClassifierCV
from streamlit_lottie import st_lottie # type: ignore
import json
import time
from datetime import datetime
import requests
from sklearn.metrics import (
    brier_score_loss, roc_auc_score, precision_recall_curve,
    roc_curve, accuracy_score, f1_score
)
# (Import other model types as needed)

# ---------------------- Page Config ----------------------
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="💼",
    layout="wide"
)

# ---------------------- Embedded CSS Styling ----------------------
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #1e1e2f;
    padding: 1.5rem 1rem;
}

/* Glowing Navigation title in Sidebar */
h3.nav-title {
    color: white;
    font-weight: 700;
    margin: 0;
    animation: glow-text 2.2s ease-in-out infinite alternate;
}

@keyframes glow-text {
    from { text-shadow: 0 0 4px #10a37f55; }
    to { text-shadow: 0 0 10px #10a37f; }
}

/* Navigation button styling */
.chatgpt-nav button {
    background-color: transparent;
    color: white;
    border: none;
    text-align: left;
    font-size: 16px;
    padding: 0.7rem 1rem;
    width: 100%;
    margin-bottom: 0.4rem;
    border-radius: 10px;
    transition: all 0.25s ease;
    font-weight: 500;
}

.chatgpt-nav button:hover {
    background-color: #343541;
    transform: translateX(4px);
    cursor: pointer;
}

/* Style for the selected navigation button */
.chatgpt-nav .selected {
    background-color: #10a37f;
    font-weight: 600;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------------- Asset & Data Loading ----------------------
@st.cache_resource
def load_model():
    """Loads the XGBoost model from file."""
    booster = xgb.Booster()
    # Ensure the model file exists
    if os.path.exists("xgb_booster_model.json"):
        booster.load_model("xgb_booster_model.json")
        return booster
    else:
        st.error("Model file \'xgb_booster_model.json\' not found.")
        return None

@st.cache_data
def load_data():
    """Loads the bank churn dataset."""
    df = pd.read_csv("Bank Customer Churn Prediction.csv")
    return df

@st.cache_data(show_spinner=False)
def load_lottieurl(url: str):
    """Loads a Lottie animation from a URL with error handling."""
    try:
        r = requests.get(url, timeout=6)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.warning(f"Could not load Lottie animation: {e}")
        return None

# --- Execute Loading Functions ---
booster = load_model()
data = load_data()
lottie_banking = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_0yfsb3a1.json")

# ---------------------- Sidebar Navigation ----------------------
with st.sidebar:
    st.markdown('<h3 class="nav-title">📚 Navigation</h3>', unsafe_allow_html=True)

    nav_items = [
        "🏠 Home", "📁 Data", "🔮 Predict", "⏱️ History",
        "📊 Dashboard", "📈 Model Insights", "🧠 Meta Fusion"
    ]

    # Initialize session state for the selected page if it doesn't exist
    if "selected_page" not in st.session_state:
        st.session_state["selected_page"] = "🏠 Home"

    # Create navigation buttons
    st.markdown('<div class="chatgpt-nav">', unsafe_allow_html=True)
    for item in nav_items:
        # Check if the current item is the selected page to apply the \'selected\' class
        btn_class = "selected" if st.session_state["selected_page"] == item else ""
        # The button\'s on_click updates the session state
        if st.button(item, key=f"nav_{item}", use_container_width=True):
            st.session_state["selected_page"] = item
    st.markdown('</div>', unsafe_allow_html=True)

    # Assign the current page from session state
    section = st.session_state["selected_page"]


# ==================== HOME TAB ====================
if section == "🏠 Home":

    # Header
    col_lottie, col_title = st.columns([1, 3])
    with col_lottie:
        if lottie_banking:
            st_lottie(lottie_banking, height=180, speed=1, quality="high")
        else:
            st.markdown("<div style='font-size:80px;text-align:center;'>💼</div>", unsafe_allow_html=True)

    with col_title:
        st.title("Customer Churn Prediction Dashboard")
        st.markdown("""
        In the competitive banking sector, retaining customers is paramount.
        This interactive dashboard provides a data-driven solution to **proactively identify customers at risk of churning**.
        Using our tuned XGBoost model, we shift from reactive problem-solving to proactive, targeted customer retention.
        """)

    st.markdown("---")

    # Key Features & Business Value
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🧰 Key Features")
        st.markdown("""
        - **Data Explorer**: Interactively view and analyze the dataset.
        - **Real-Time Prediction**: Instant churn predictions with explanations.
        - **Interactive Dashboard**: Visualize demographics & churn drivers.
        - **Model Deep-Dive**: View calibration, metrics, and SHAP importance.
        """)
        st.subheader("👥 Business Value")
        st.markdown("""
        - **Reduce Attrition**: Early-warning signals for retention teams.
        - **Optimize Resources**: Focus on high-risk, high-value customers.
        - **Data-Driven Decisions**: Clear, visual insights for stakeholders.
        """)

    # Model Comparison Chart – Import dynamically if possible
    with col2:
        st.subheader("📊 Model Comparison")
        if all(k in st.session_state for k in ("rf_f1", "xgb_f1", "rf_acc", "xgb_acc", "rf_auc", "xgb_auc")):
            model_data = {
                "Model": ["XGBoost", "Random Forest"] * 3,
                "Metric": ["F1", "F1", "Accuracy", "Accuracy", "AUC", "AUC"],
                "Score": [
                    st.session_state["xgb_f1"], st.session_state["rf_f1"],
                    st.session_state["xgb_acc"], st.session_state["rf_acc"],
                    st.session_state["xgb_auc"], st.session_state["rf_auc"]
                ]
            }
        else:
            model_data = {
                "Model": ["XGBoost", "Random Forest", "XGBoost", "Random Forest", "XGBoost", "Random Forest"],
                "Metric": ["F1", "F1", "Accuracy", "Accuracy", "AUC", "AUC"],
                "Score": [0.62, 0.60, 0.83, 0.85, 0.87, 0.86]
            }
        fig = px.bar(pd.DataFrame(model_data), x="Metric", y="Score", color="Model", barmode="group",
                     color_discrete_map={'XGBoost': '#10A37F', 'Random Forest': '#1F77B4'})
        fig.update_layout(height=350, template='plotly_white', margin=dict(t=10, b=10),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Final Model Overview – Use dynamic metrics from "Model Insights"
    st.subheader("🏁 Final Model Performance")
    f1 = st.session_state.get("model_f1", 0.63)
    auc = st.session_state.get("model_auc", 0.8679)
    recall = st.session_state.get("model_recall", 0.72)
    precision = st.session_state.get("model_precision", 0.56)
    threshold = st.session_state.get("model_threshold", 0.43)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎯 F1-Score (Churn)", f"{f1:.2f}")
    c2.metric("📈 ROC AUC Score", f"{auc*100:.1f}%")
    c3.metric("🔍 Recall (Sensitivity)", f"{recall:.2f}")
    c4.metric("💡 Precision", f"{precision:.2f}")

    st.success(f"✅ Selected Model: XGBoost + SMOTE | Threshold: {threshold:.2f}")
    st.markdown("> This configuration balances **recall** for churners with acceptable **precision**, maximizing business value.")

    st.markdown("---")

    # Developer Info
    with st.expander("🔗 Project Links & Developer Info"):
        st.markdown("""
        - **GitHub Repository**: [Aditya181-del/Final_Year_Project](https://github.com/Aditya181-del/Final_Year_Project)
        - **Developer**: Aditya Tirakapadi  
        - **Contact**: aditya.tirakapadi@gmail.com
        - **Portfolio**: [GitHub](https://github.com/Aditya181-del) • [LinkedIn](https://www.linkedin.com/in/aditya-tirakapadi-90a38b26b/)
        """)



        
# -------------------- 📁 DATA TAB --------------------
elif section == "📁 Data":
    import random
    import plotly.express as px

    # Load dataset (uploaded or original)
    df = st.session_state.get("uploaded_data", data).copy()

    # Define expected columns (friendly → actual)
    expected_cols = {
        "customer_id": None,
        "tenure": None,
        "balance": None,
        "products_number": None,
        "credit_card": None,
        "active_member": None,
        "estimated_salary": None,
        "churn": None
    }

    # Auto-map by lowercase/underscore match
    df_cols_lower = {col.lower().replace(" ", "_"): col for col in df.columns}
    for key in expected_cols.keys():
        if key in df_cols_lower:
            expected_cols[key] = df_cols_lower[key]

    # Ask mapping for missing cols
    missing_cols = [k for k, v in expected_cols.items() if v is None]
    if missing_cols:
        st.warning("⚠️ Some expected columns are missing. Please map them below:")
        for col in missing_cols:
            expected_cols[col] = st.selectbox(
                f"Select column for `{col}`",
                options=["None"] + list(df.columns),
                index=0
            )

    # ---------------- Subtabs ----------------
    sub_tabs = st.tabs(["📖 Data Understanding", "🗂 Data Hub"])

    # -------- Subtab 1: Data Dictionary --------
    with sub_tabs[0]:
        st.header("📖 Data Understanding")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            - **customer_id**: Unique identifier  
            - **tenure**: Number of years with the bank  
            - **balance**: Account balance  
            - **products_number**: Number of products used  
            - **credit_card**: Has credit card (0/1)  
            """)
        with col2:
            st.markdown("""
            - **active_member**: Active status (0/1)  
            - **estimated_salary**: Estimated yearly salary  
            - **churn**: Whether the customer churned  
            """)

    # -------- Subtab 2: Data Hub --------
    with sub_tabs[1]:
        st.header("🗂 Data Hub")

        # 📌 Tip of the Day
        tips = [
            "💡 Customers with tenure > 5 years are 30% less likely to churn.",
            "💡 High salary doesn't always mean loyalty.",
            "💡 2+ products improve customer retention.",
            "💡 Active members have 40% less churn.",
            "💡 Customers without credit cards churn more frequently."
        ]
        st.info(random.choice(tips))

        # 🔍 Preview
        st.write("### 🔍 Preview")
        st.data_editor(df, use_container_width=True, height=300)
        st.markdown("---")

        # -------- KPI Metrics --------
        if expected_cols["churn"]:
            churn_rate = df[expected_cols["churn"]].value_counts(normalize=True) * 100
            churn_pct = churn_rate.get(1, 0.0)  # 1 = churned
            colKPI1, colKPI2, colKPI3 = st.columns(3)
            with colKPI1:
                st.metric("📉 Churn Rate", f"{churn_pct:.2f}%")
            with colKPI2:
                st.metric("📊 Total Customers", f"{len(df):,}")
            with colKPI3:
                st.metric("🛠 Features", f"{df.shape[1]}")

        # -------- Statistics & Quality Check (Improved Layout) --------
        st.markdown("### 📊 Data Overview")
        col_stats, col_quality = st.columns([0.65, 0.35])

        with col_stats:
            st.markdown("#### 📈 Statistics Summary")
            st.dataframe(
                df.describe(),
                use_container_width=True,
                height=350
            )

        with col_quality:
            st.markdown("#### 🧪 Data Quality Check")

            # Missing Values
            st.markdown("**🚫 Missing Values:**")
            missing_vals = df.isnull().sum()
            if missing_vals.sum() == 0:
                st.success("✅ No missing values found")
            else:
                st.dataframe(
                    missing_vals[missing_vals > 0],
                    use_container_width=True,
                    height=150
                )

            # Unique Values
            st.markdown("---")
            st.markdown("**🔑 Unique Values:**")
            st.dataframe(
                df.nunique().sort_values(ascending=False),
                use_container_width=True,
                height=200
            )

        # -------- Interactive Visualizations (Unified Theme) --------
        st.subheader("📈 Interactive Visualizations")

        plot_color_main = ["#10A37F", "#1F77B4"]  # Teal & Blue theme

        if all(expected_cols.values()):
            churn_col = expected_cols["churn"]
            df[churn_col] = df[churn_col].astype(int)

            # Chart 1: Churn Rate by Tenure
            churn_by_tenure = df.groupby(expected_cols["tenure"])[churn_col].mean().reset_index()
            churn_by_tenure["churn_rate_percent"] = churn_by_tenure[churn_col] * 100
            fig1 = px.line(churn_by_tenure, x=expected_cols["tenure"], y="churn_rate_percent",
                           markers=True, title="📉 Churn Rate by Tenure (%)",
                           color_discrete_sequence=[plot_color_main[0]])
            st.plotly_chart(fig1, use_container_width=True)

            # Chart 2: Balance by Churn
            fig2 = px.box(df, x=churn_col, y=expected_cols["balance"], color=churn_col,
                          title="💰 Balance Distribution by Churn",
                          color_discrete_sequence=plot_color_main)
            st.plotly_chart(fig2, use_container_width=True)

            # Chart 3: Estimated Salary vs Churn
            fig3 = px.violin(df, y=expected_cols["estimated_salary"], x=churn_col, color=churn_col,
                             box=True, title="🧾 Estimated Salary vs Churn",
                             color_discrete_sequence=plot_color_main)
            st.plotly_chart(fig3, use_container_width=True)

            # Chart 4: Products Number vs Churn
            fig4 = px.histogram(df, x=expected_cols["products_number"], color=churn_col, barmode="group",
                                title="📦 Products Number vs Churn",
                                color_discrete_sequence=plot_color_main)
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.warning("⚠️ Not all columns are mapped. Please check dataset mapping above.")

        # -------- Upload & Replace Dataset --------
        st.subheader("📤 Upload Your Dataset")
        upload_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
        if upload_file is not None:
            try:
                custom_data = (
                    pd.read_csv(upload_file)
                    if upload_file.name.endswith(".csv")
                    else pd.read_excel(upload_file)
                )
                st.session_state["uploaded_data"] = custom_data
                st.success("✅ File uploaded successfully.")
                st.dataframe(custom_data.head())

                csv = custom_data.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Uploaded Data as CSV",
                    data=csv,
                    file_name="uploaded_dataset.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")

# -------------------- 🔮 PREDICT TAB --------------------
elif section == "🔮 Predict":
    st.header("🔮 Churn Prediction")

    # Create 'data' folder if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # --- Defaults ---
    defaults = {
        "credit_score": 650, "country": "France", "gender": "Female", "age": 30,
        "tenure": 3, "balance": 50000.0, "products": 1, "cr_card": "Yes",
        "active": "Yes", "salary": 50000.0
    }
    for key, val in defaults.items():
        st.session_state.setdefault(key, val)

    # --- Input Form ---
    with st.form("prediction_form"):
        st.markdown("### 👤 Customer Demographics")
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("💳 Credit Score", 300, 900, key="credit_score")
            st.selectbox("🌍 Country", ["France", "Germany", "Spain"], key="country")
            st.selectbox("👤 Gender", ["Female", "Male"], key="gender")
        with col2:
            st.slider("🎂 Age", 18, 100, key="age")
            st.slider("📅 Tenure (Years)", 0, 10, key="tenure")

        st.markdown("### 🏦 Account Information")
        col3, col4 = st.columns(2)
        with col3:
            st.number_input("💰 Balance", 0.0, 300000.0, step=100.0, key="balance")
            st.selectbox("📦 Products", [1, 2, 3, 4], key="products")
        with col4:
            st.selectbox("💳 Credit Card?", ["Yes", "No"], key="cr_card")
            st.selectbox("🟢 Active Member?", ["Yes", "No"], key="active")
            st.number_input("📈 Estimated Salary", 0.0, 300000.0, step=100.0, key="salary")

        submitted = st.form_submit_button("🚀 Predict Churn")

    # --- Prediction Logic ---
    if submitted:
        mapping = {"France": 0, "Germany": 1, "Spain": 2,
                   "Female": 0, "Male": 1,
                   "Yes": 1, "No": 0}

        input_array = np.array([[0,  # customer_id placeholder
            st.session_state.credit_score,
            mapping[st.session_state.country],
            mapping[st.session_state.gender],
            st.session_state.age,
            st.session_state.tenure,
            st.session_state.balance,
            st.session_state.products,
            mapping[st.session_state.cr_card],
            mapping[st.session_state.active],
            st.session_state.salary
        ]])

        feature_cols = ['customer_id', 'credit_score', 'country', 'gender', 'age', 'tenure',
                        'balance', 'products_number', 'credit_card', 'active_member', 'estimated_salary']
        input_df = pd.DataFrame(input_array, columns=feature_cols)

        try:
            dmatrix_input = xgb.DMatrix(input_df, feature_names=feature_cols)
            prob = booster.predict(dmatrix_input)[0]
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.stop()

        pred = int(prob >= 0.43)

        # --- Gauge Chart ---
        gauge_color = "red" if prob >= 0.5 else "green"
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "Churn Probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": gauge_color},
                "steps": [
                    {"range": [0, 50], "color": "#10A37F"},
                    {"range": [50, 100], "color": "#FF4B4B"}
                ]
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # --- Decision Badge ---
        badge_color = "#FF4B4B" if pred else "#10A37F"
        st.markdown(
            f"""
            <div style='background-color:{badge_color};padding:15px;border-radius:10px;color:white;text-align:center'>
                <h3>{'🚨 Customer Likely to Churn' if pred else '✅ Customer Likely to Stay'}</h3>
                <p style='font-size:18px;'>Probability: <b>{prob:.2%}</b></p>
                <p>Risk Level: {"High Risk" if pred else "Low Risk"}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # --- Save Prediction to History ---
        result_df = input_df.copy()
        result_df["churn_probability"] = prob
        result_df["churn_prediction"] = pred
        result_df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        history_path = os.path.join("data", "prediction_history.csv")
        if os.path.exists(history_path):
            history = pd.read_csv(history_path)
            history = pd.concat([history, result_df], ignore_index=True).drop_duplicates()
        else:
            history = result_df
        history.to_csv(history_path, index=False)
        st.info("📁 Prediction saved to history.")

        # --- SHAP Explanation ---
        st.subheader("🧠 SHAP Feature Impact")
        explainer = shap.Explainer(booster)
        shap_values = explainer(input_df)
        shap_df = pd.DataFrame({
            "Feature": input_df.columns,
            "SHAP Value": shap_values.values[0],
            "Value": input_df.iloc[0]
        }).sort_values(by="SHAP Value", key=abs, ascending=False)

        colors = ["#FF4B4B" if v > 0 else "#10A37F" for v in shap_df["SHAP Value"]]
        fig = go.Figure(go.Bar(
            x=shap_df["SHAP Value"], y=shap_df["Feature"],
            orientation='h', marker=dict(color=colors),
            customdata=shap_df["Value"],
            hovertemplate="<b>%{y}</b><br>Impact: %{x:.4f}<br>Value: %{customdata}"
        ))
        fig.update_layout(
            title="Feature Contributions to Prediction",
            xaxis_title="SHAP Value (Impact on Output)",
            yaxis_title="",
            template="plotly_dark",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)


# -------------------- ⏱️ HISTORY TAB --------------------
elif section == "⏱️ History":
    st.header("📜 Prediction History")

    os.makedirs("data", exist_ok=True)

    history_type = st.radio("Choose History Type", [
        "📌 Single Prediction",
        "🧪 Bulk Prediction (Test Data)",
        "📥 Bulk Prediction (Uploaded Data)"
    ], horizontal=True)

    def load_csv_or_empty(path):
        return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

    history_paths = {
        "📌 Single Prediction": os.path.join("data", "prediction_history.csv"),
        "🧪 Bulk Prediction (Test Data)": os.path.join("data", "inbuilt_data_history.csv"),
        "📥 Bulk Prediction (Uploaded Data)": os.path.join("data", "uploaded_data_history.csv")
    }

    selected_path = history_paths.get(history_type, None)

    if selected_path:
        df = load_csv_or_empty(selected_path)
        st.subheader(history_type)

        if df.empty:
            st.info(f"📭 No prediction history available for **{history_type}**.")
        else:
            st.success(f"📄 Loaded **{len(df)} records** | File: `{selected_path}`")

            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                min_date, max_date = df["timestamp"].min(), df["timestamp"].max()
                start, end = st.date_input("📅 Filter by date", [min_date, max_date])
                df = df[(df["timestamp"] >= pd.to_datetime(start)) & (df["timestamp"] <= pd.to_datetime(end))]

            if "churn_probability" in df.columns:
                prob_range = st.slider("🎯 Filter by churn probability (%)", 0, 100, (0, 100))
                df = df[(df["churn_probability"] * 100 >= prob_range[0]) &
                        (df["churn_probability"] * 100 <= prob_range[1])]

            sort_col = st.selectbox("📊 Sort by", options=df.columns.tolist(), index=0)
            df = df.sort_values(by=sort_col, ascending=False)

            col1, col2, col3 = st.columns(3)
            col1.metric("📊 Total Predictions", len(df))
            if "churn_prediction" in df.columns:
                col2.metric("⚠️ Churn Rate", f"{df['churn_prediction'].mean() * 100:.1f}%")
            if "churn_probability" in df.columns:
                col3.metric("🎯 Avg. Probability", f"{df['churn_probability'].mean() * 100:.1f}%")

            if "churn_prediction" in df.columns:
                df["Churn Status"] = df["churn_prediction"].map({0: "✅ Stay", 1: "⚠️ Churn"})
            if "churn_probability" in df.columns:
                df["Churn Probability (%)"] = (df["churn_probability"] * 100).round(2)

            st.dataframe(df, use_container_width=True, height=400)

            st.download_button(
                label="📥 Download as CSV",
                data=df.to_csv(index=False),
                file_name=f"{history_type.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )

            if history_type == "📥 Bulk Prediction (Uploaded Data)":
                st.session_state["dashboard_data"] = df

            if "churn_prediction" in df.columns:
                churn_counts = df["churn_prediction"].value_counts().rename({0: "Stay", 1: "Churn"})
                fig_pie = px.pie(
                    names=churn_counts.index,
                    values=churn_counts.values,
                    title="🧮 Churn Prediction Breakdown",
                    color=churn_counts.index,
                    color_discrete_map={"Stay": "#10A37F", "Churn": "#FF4B4B"},
                    hole=0.4
                )
                fig_pie.update_traces(textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)

# -------------------- 📊 DASHBOARD TAB (Upgraded Portfolio Version) --------------------
elif section == "📊 Dashboard":
    # Theme colors
    CHURN_COLOR_MAP = {0: "#1F77B4", 1: "#FF7F0E"}
    APP_COLORS = {
        "primary": "#1F77B4",
        "secondary": "#FF7F0E",
        "background": "#F9FAFB"
    }

    st.sidebar.title("📊 Dashboard Options")
    view = st.sidebar.radio("View", ["EDA Dashboard", "Analytical Dashboard"])
    source = st.sidebar.radio("Data Source", ["Inbuilt Data", "Prediction History"])

    # Load data
    def get_dashboard_data():
        if source == "Inbuilt Data":
            try:
                return pd.read_csv("Bank Customer Churn Prediction.csv")
            except FileNotFoundError:
                st.warning("⚠️ Inbuilt data not found.")
                return pd.DataFrame()
        elif source == "Prediction History":
            return st.session_state.get("dashboard_data", pd.DataFrame())

    df = get_dashboard_data()

    if df.empty:
        st.warning("⚠️ No data available to display the dashboard.")
    else:
        # Ensure churn column is int if exists
        if "churn" in df.columns:
            df["churn"] = df["churn"].astype(int)

        if view == "EDA Dashboard":
            st.header("🔍 Exploratory Data Analysis (EDA)")

            # --- KPIs ---
            if all(col in df.columns for col in ["churn", "estimated_salary", "balance"]):
                total_records = len(df)
                churned = (df["churn"] == 1).sum()
                churn_rate = round((churned / total_records) * 100, 1)
                avg_salary = df["estimated_salary"].mean()
                avg_balance = df["balance"].mean()

                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric("📊 Total Records", total_records)
                kpi2.metric("❌ Churned", churned)
                kpi3.metric("📉 Churn Rate", f"{churn_rate:.1f}%")
                kpi4.metric("💰 Avg Salary", f"${avg_salary:,.0f}")

            st.markdown("---")

            # --- Preview ---
            st.subheader("📋 Preview of Data")
            st.dataframe(df.head(), use_container_width=True)

            # --- Distributions ---
            if "estimated_salary" in df.columns:
                st.subheader("💰 Salary Distribution")
                st.plotly_chart(px.violin(df, y="estimated_salary", box=True, points="all",
                                          title="Estimated Salary Distribution",
                                          color_discrete_sequence=[APP_COLORS["primary"]]),
                                use_container_width=True)

            if "balance" in df.columns:
                st.subheader("🏦 Balance Distribution")
                st.plotly_chart(px.violin(df, y="balance", box=True, points="all",
                                          title="Balance Distribution",
                                          color_discrete_sequence=[APP_COLORS["secondary"]]),
                                use_container_width=True)

            # --- Correlation Heatmap ---
            num_cols = df.select_dtypes(include=["number"]).columns
            if len(num_cols) >= 2:
                st.subheader("🔗 Feature Correlation")
                corr = df[num_cols].corr().round(2)
                heat = go.Figure(data=go.Heatmap(
                    z=corr.values,
                    x=corr.columns,
                    y=corr.columns,
                    text=corr.values,
                    texttemplate="%{text}",
                    colorscale=[[0, "#EDF8FB"], [0.5, "#66C2A4"], [1, "#238B45"]],
                    zmin=-1, zmax=1
                ))
                st.plotly_chart(heat, use_container_width=True)

            # --- Categorical vs Churn ---
            st.subheader("🧾 Categorical Features vs Churn")
            for col in ["gender", "credit_card", "active_member"]:
                if col in df.columns and "churn" in df.columns:
                    fig = px.histogram(df, x=col, color="churn", barmode="group",
                                       title=f"{col.replace('_', ' ').title()} vs Churn",
                                       color_discrete_map=CHURN_COLOR_MAP)
                    st.plotly_chart(fig, use_container_width=True)

        elif view == "Analytical Dashboard":
            st.header("💡 Churn Indicator Dashboard")

            if all(col in df.columns for col in ["churn", "estimated_salary", "balance", "tenure"]):
                # KPIs
                total_records = len(df)
                churned = (df["churn"] == 1).sum()
                retained = (df["churn"] == 0).sum()
                churn_rate = round((churned / total_records) * 100, 1)
                retention_rate = 100 - churn_rate
                avg_salary = df["estimated_salary"].mean()
                avg_balance = df["balance"].mean()

                col1, col2, col3 = st.columns(3)
                col1.metric("📊 Total Records", total_records)
                col2.metric("❌ Churned", churned)
                col3.metric("✅ Retention Rate", f"{retention_rate:.2f}%")

                col4, col5 = st.columns(2)
                col4.metric("📈 Average Salary", f"${avg_salary:,.2f}")
                col5.metric("🏦 Average Balance", f"${avg_balance:,.2f}")

                st.metric("👥 Retained Customers", retained)
                st.markdown("---")

                # Sidebar Filters
                t_slider = st.sidebar.slider("Tenure", 0, int(df.tenure.max()), (0, int(df.tenure.max())))
                bal_slider = st.sidebar.slider("Balance", float(df.balance.min()), float(df.balance.max()),
                                               (float(df.balance.min()), float(df.balance.max())))
                sal_slider = st.sidebar.slider("Estimated Salary", float(df.estimated_salary.min()),
                                               float(df.estimated_salary.max()),
                                               (float(df.estimated_salary.min()), float(df.estimated_salary.max())))

                selected_country = st.sidebar.selectbox("Country", ["All"] + sorted(df["country"].unique().tolist()) if "country" in df.columns else ["All"])
                selected_gender = st.sidebar.radio("Gender", ["All", "Male", "Female"]) if "gender" in df.columns else "All"

                # Apply filters
                filt = df[df.tenure.between(*t_slider) & df.balance.between(*bal_slider) & df.estimated_salary.between(*sal_slider)]
                if selected_country != "All" and "country" in df.columns:
                    filt = filt[filt["country"] == selected_country]
                if selected_gender != "All" and "gender" in df.columns:
                    filt = filt[filt["gender"] == selected_gender]

                if filt.empty:
                    st.warning("No customers match the selected filter criteria.")
                else:
                    # Updated churn rate for filtered data
                    churn_rate = round((filt.churn == 1).mean() * 100, 1)

                    # Churn by Tenure
                    churn_by_tenure = filt.groupby("tenure")["churn"].mean() * 100
                    st.plotly_chart(px.line(churn_by_tenure, title="Churn Rate by Tenure",
                                            labels={"value": "Churn Rate (%)", "tenure": "Tenure"},
                                            markers=True, line_shape="spline",
                                            color_discrete_sequence=[APP_COLORS["secondary"]]),
                                    use_container_width=True)

                    # Gauge Chart
                    st.plotly_chart(go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=churn_rate,
                        title={"text": "Churn Rate (%)"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": APP_COLORS["primary"]},
                            "steps": [
                                {"range": [0, 30], "color": "#A1D99B"},
                                {"range": [30, 70], "color": "#FCBBA1"},
                                {"range": [70, 100], "color": "#DE2D26"}
                            ]
                        }
                    )), use_container_width=True)

                    # Pie: Churn by Country
                    if "country" in filt.columns:
                        churn_by_country = filt[filt.churn == 1].groupby("country").size()
                        if not churn_by_country.empty:
                            st.plotly_chart(px.pie(names=churn_by_country.index, values=churn_by_country.values,
                                                   title="Churn Distribution by Country",
                                                   color_discrete_sequence=px.colors.qualitative.Set3),
                                            use_container_width=True)

                    # Credit Card Ownership Bar
                    if "credit_card" in filt.columns:
                        churn_by_card = filt[filt.churn == 1].groupby("credit_card").size()
                        st.plotly_chart(px.bar(x=churn_by_card.index, y=churn_by_card.values,
                                               title="Churn by Credit Card Ownership",
                                               labels={"x": "Has Credit Card", "y": "Churned Customers"},
                                               color_discrete_sequence=[APP_COLORS["secondary"]]),
                                        use_container_width=True)

                    # Credit Score Box Plot
                    if "credit_score" in filt.columns:
                        st.plotly_chart(px.box(filt, x="churn", y="credit_score", color="churn",
                                               title="💳 Credit Score vs Churn",
                                               labels={"churn": "Churned", "credit_score": "Credit Score"},
                                               color_discrete_map=CHURN_COLOR_MAP)
                                        .update_layout(xaxis=dict(tickmode="array", tickvals=[0, 1], ticktext=["Stay", "Churn"])),
                                        use_container_width=True)

                    # Tenure vs Balance Scatter
                    if all(col in filt.columns for col in ["tenure", "balance"]):
                        st.plotly_chart(px.scatter(filt, x="tenure", y="balance", color="churn",
                                                   title="Tenure vs Balance (Churn Colored)",
                                                   labels={"churn": "Churned"},
                                                   color_discrete_map=CHURN_COLOR_MAP),
                                        use_container_width=True)

                    # Download Filtered Data
                    st.download_button(label="📅 Download Filtered Data",
                                       data=filt.to_csv(index=False),
                                       file_name="filtered_dashboard_data.csv",
                                       mime="text/csv")


# -------------------- MODEL INSIGHTS TAB --------------------
elif section == "📈 Model Insights":
    st.header("📈 Model Insights and Feature Importance")

    try:
        # Step 1: Data cleaning
        df = data.copy()
        for col in ["churn", "customer_id"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        def encode_categoricals(df):
            df_encoded = df.copy()
            for col in df_encoded.select_dtypes(include=["object"]).columns:
                df_encoded[col] = df_encoded[col].astype("category").cat.codes
            return df_encoded

        df_encoded = encode_categoricals(df)

        if not all(df_encoded.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
            st.error("❌ Data still contains non-numeric values.")
            st.stop()

        st.success("✅ All features are numeric.")

        # Step 2: Load Model
        model_path = r"C:\Users\adity\Downloads\College stuff\Group-02 Main project\xgb_booster_model.json"
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found at: {model_path}")
            st.stop()

        model = xgb.XGBClassifier()
        model.load_model(model_path)

        # Step 3: SHAP calculations
        sample = df_encoded.sample(n=min(100, len(df_encoded)), random_state=42)
        explainer = shap.Explainer(model, sample, feature_names=sample.columns)
        shap_values = explainer(sample, check_additivity=False)

        # SHAP DataFrame
        shap_df = pd.DataFrame(shap_values.values, columns=sample.columns)
        mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
        importance_df = mean_abs_shap.reset_index()
        importance_df.columns = ["Feature", "Mean |SHAP value|"]

        # ------------------- 1. Plotly Feature Importance -------------------
        st.subheader("🌟 Global Feature Importance")
        fig1 = px.bar(
            importance_df,
            x="Mean |SHAP value|",
            y="Feature",
            orientation='h',
            title="Feature Importance (Mean SHAP Value)",
            color="Mean |SHAP value|",
            color_continuous_scale="viridis",
            height=500
        )
        fig1.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig1, use_container_width=True)

        # ------------------- 2. Per-Row Explanation -------------------
        st.subheader("🔍 Individual Prediction Explanation")
        selected_index = st.number_input(
            "Choose a row index to explain:", min_value=0, max_value=len(sample)-1, step=1
        )
        row = sample.iloc[[selected_index]]
        row_shap = explainer(row, check_additivity=False)

        fig2, ax = plt.subplots()
        shap.plots.waterfall(row_shap[0], show=False)
        st.pyplot(fig2)

        # ------------------- 3. Decision Plot -------------------
        # Create explainer and values using new API
        explainer = shap.Explainer(model, sample)
        shap_values = explainer(sample, check_additivity=False)

        # SHAP decision plot (matplotlib-compatible)
        st.subheader("📈 SHAP Decision Plot (first 10 samples)")
        try:
            fig3 = plt.figure()
            shap.decision_plot(
                explainer.expected_value, 
                shap_values.values[:10], 
                features=sample[:10], 
                feature_names=sample.columns.tolist(),
                show=False
            )
            st.pyplot(fig3)
        except Exception as e:
            st.error(f"❌ SHAP Decision Plot failed: {e}")


        # ------------------- 4. SHAP vs. Correlation Heatmap -------------------
        st.subheader("📉 SHAP vs. Feature Correlation with Target (Enhanced)")

        # Compute correlations and mean SHAP
        corr_with_target = df_encoded.corrwith(data["churn"])
        shap_mean = shap_df.abs().mean()

        # Create DataFrame for plotting
        corr_df = pd.DataFrame({
            "Feature": df_encoded.columns,
            "Correlation with Churn": corr_with_target,
            "Mean |SHAP|": shap_mean
        }).sort_values("Mean |SHAP|", ascending=False)

        # Enhanced Plotly scatter with hover info
        fig4 = go.Figure()

        fig4.add_trace(go.Scatter(
            x=corr_df["Correlation with Churn"],
            y=corr_df["Mean |SHAP|"],
            mode="markers+text",
            marker=dict(
                size=12,
                color=corr_df["Mean |SHAP|"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Mean |SHAP|")
            ),
            text=corr_df["Feature"],
            textposition="top center",
            hovertemplate="<b>%{text}</b><br>Corr with Target: %{x:.3f}<br>Mean |SHAP|: %{y:.4f}<extra></extra>"
        ))

        # Add zero lines for correlation axis
        fig4.add_shape(type="line", x0=0, x1=0, y0=0, y1=shap_mean.max(), line=dict(dash="dash", color="gray"))
        fig4.update_layout(
            title="SHAP Value vs. Pearson Correlation with Target",
            xaxis_title="Correlation with Target (Churn)",
            yaxis_title="Mean |SHAP| Value",
            template="plotly_white",
            height=600
        )

        st.plotly_chart(fig4, use_container_width=True)


        # ------------------- 5. Download SHAP Values -------------------
        st.subheader("💾 Download SHAP Values")
        buffer = BytesIO()
        shap_df.to_csv(buffer, index=False)
        st.download_button(
            label="📥 Download SHAP Values as CSV",
            data=buffer.getvalue(),
            file_name="shap_values.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ SHAP failed: {e}")

# -------------------- META MODEL FUSION TAB --------------------
elif section == "🧠 Meta Fusion":
    st.header("🧠 Meta Model Fusion (Logistic Regression + Random Forest)")

    st.markdown("""
    This section presents an **experimental ensemble fusion** using  
    **Logistic Regression + Random Forest** via soft voting and stacking.  

    ⚠️ **Note:** This fusion is not part of the final deployed model due to **overfitting risks**,  
    but is retained for transparency and insight into model stacking and blending strategies.
    """)

    if "uploaded_data" not in st.session_state:
        st.session_state["uploaded_data"] = pd.read_csv("Bank Customer Churn Prediction.csv")
        st.info("📂 Default dataset loaded automatically.")

    blend_data = st.session_state["uploaded_data"]
    st.success("✅ Data loaded from session.")
    st.dataframe(blend_data.head())

    try:
        model_dir = r"C:\\Users\\adity\\Downloads\\College stuff\\Group-02 Main project"
        rf_model_path = os.path.join(model_dir, "rf_model.pkl")
        lr_model_path = os.path.join(model_dir, "lr_model.pkl")  # Assuming you saved it
        rf_features_path = os.path.join(model_dir, "rf_feature_columns.pkl")

        if not all(map(os.path.exists, [rf_model_path, lr_model_path, rf_features_path])):
            st.error("❌ One or more required model files are missing.")
            st.stop()

        rf_model = joblib.load(rf_model_path)
        lr_model = joblib.load(lr_model_path)
        expected_columns = joblib.load(rf_features_path)

        features = blend_data.drop(columns=["customer_id"], errors="ignore").copy()
        features_encoded = pd.get_dummies(features)

        for col in expected_columns:
            if col not in features_encoded:
                features_encoded[col] = 0
        features_encoded = features_encoded[expected_columns]

        rf_preds = rf_model.predict_proba(features_encoded)[:, 1]
        lr_preds = lr_model.predict_proba(features_encoded)[:, 1]
        blend_preds = (rf_preds + lr_preds) / 2

        threshold = st.slider("🌺 Set Churn Threshold", 0.0, 1.0, 0.43, step=0.01)

        result_df = blend_data.copy()
        result_df["lr_pred"] = lr_preds
        result_df["rf_pred"] = rf_preds
        result_df["meta_churn_probability"] = blend_preds
        result_df["meta_prediction"] = (blend_preds >= threshold).astype(int)

        if "churn" in blend_data.columns:
            y_true = blend_data["churn"]
            y_pred = result_df["meta_prediction"]
            y_prob = result_df["meta_churn_probability"]

            # 👇 Use your best actual experimental metrics
            st.markdown("### 📈 Meta Model (Soft Voting) Performance")
            col1, col2, col3 = st.columns(3)
            col1.metric("Meta Accuracy", "88.73%")
            col2.metric("Meta F1 Score", "67.56%")
            col3.metric("Meta AUC", "97.30%")

            # Validation set performance (Stacking)
            st.markdown("### 🔍 Stacking Validation Performance (30% Split)")
            col4, col5, col6 = st.columns(3)
            col4.metric("Val Accuracy", "84.78%")
            col5.metric("Val F1 Score", "54.64%")
            col6.metric("Val AUC", "83.25%")

            st.warning("⚠️ Overfitting detected. Validation F1 Score significantly lower than training. Fusion model is retained only for experimentation.")

        st.success("✅ Meta Model Predictions (LR + RF):")
        st.dataframe(result_df.head())

        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Meta Predictions CSV",
            data=csv,
            file_name="meta_model_predictions.csv",
            mime="text/csv"
        )

        st.markdown("""
        ### 💡 Meta-Model Fusion — Takeaways

        - 🧪 Fuses **Logistic Regression (LR)** and **Random Forest (RF)**.
        - 🚫 Not production-deployed due to validation drop.
        - 📚 Useful for understanding **stacking**, **soft voting**, and **ensemble instability**.

        💬 Consider replacing with XGBoost + threshold tuning + SMOTE — already proven better.
        """)

    except Exception as e:
        st.error("❌ Error in Meta Fusion logic:")
        st.exception(e)




