# combined_churn_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# -------------------------
# Page config + CSS
# -------------------------
st.set_page_config(page_title="Telecom Churn — Login · Predict · Dashboard",
                   page_icon="📡",
                   layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(120deg, #07102a 0%, #0a1a2b 60%, #07102a 100%); color: #eaf3ff; }
.glass { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.06); border-radius:12px; padding:14px; box-shadow: 0 8px 30px rgba(2,6,23,0.6); backdrop-filter: blur(6px); }
.kpi { padding:12px; border-radius:10px; text-align:left; }
.kpi .label { font-size:0.95rem; color:#bcd7ff; }
.kpi .value { font-size:1.6rem; font-weight:700; color:#ffffff; }
.explain { color:#cbd7e6; font-size:0.95rem; margin-top:10px; }
.header { font-size:26px; font-weight:700; background: linear-gradient(90deg,#7ee6ff,#6e8cff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.small-muted { color:#9fb3d6; font-size:0.9rem; }
.card { background: rgba(255,255,255,0.03); padding:12px; border-radius:10px; }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Paths for model and encoders
# -------------------------
MODEL_PATH = "customer_churn_model.pkl"
ENCODERS_PATH = "encoders.pkl"

# -------------------------
# Login system
# -------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔒 Telecom Churn — Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        # simple demo creds (change for production)
        if username == "sravalika" and password == "12345678":
            st.session_state.logged_in = True
            st.success("✅ Login successful — loading the app...")
            st.rerun()
            st.error("❌ Invalid username or password")
    st.stop()
else:
    st.sidebar.success("✅ Logged in")

# -------------------------
# Utility: safe label encode (append unseen labels)
# -------------------------
def safe_transform_label(le, value):
    """
    Transform a single value using sklearn LabelEncoder le.
    If value not in classes_, append it so transform won't fail.
    Returns integer encoded label.
    """
    try:
        # if classes_ doesn't contain the value, append it
        if value not in le.classes_:
            le.classes_ = np.append(le.classes_, value)
        return int(le.transform([value])[0])
    except Exception:
        # fallback: attempt to map by string
        try:
            mapping = {c: i for i, c in enumerate(le.classes_)}
            return mapping.get(value, 0)
        except Exception:
            return 0

# -------------------------
# Load model & encoders (if present)
# -------------------------
model = None
label_encoders = {}

if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODERS_PATH):
    st.warning("Model or encoders not found in app folder. Prediction page will be disabled until you upload them.")
else:
    try:
        with open(MODEL_PATH, "rb") as f:
            model_loaded = pickle.load(f)
            model = model_loaded['model'] if isinstance(model_loaded, dict) and 'model' in model_loaded else model_loaded
    except Exception as e:
        st.error(f"Error loading model: {e}")
        model = None

    try:
        with open(ENCODERS_PATH, "rb") as f:
            label_encoders = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading encoders: {e}")
        label_encoders = {}

# -------------------------
# Sidebar navigation
# -------------------------
page = st.sidebar.selectbox("Navigate to", ["Prediction", "Dashboard"])

# -------------------------
# Prediction Page
# -------------------------
if page == "Prediction":
    st.markdown('<div class="header">Churn Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-muted">Enter customer data to predict churn probability.</div>', unsafe_allow_html=True)
    st.markdown("")

    if model is None or not label_encoders:
        st.error("Prediction model or encoders not loaded. Go to *Upload model/encoders* in the sidebar to upload files (customer_churn_model.pkl and encoders.pkl).")
    else:
        # form inputs
        with st.form("churn_form"):
            st.subheader("Customer attributes")
            cols = st.columns(3)
            gender = cols[0].selectbox("Gender", ["Male", "Female"])
            senior = cols[1].selectbox("Senior Citizen (0/1)", [0, 1])
            partner = cols[2].selectbox("Partner", ["Yes", "No"])

            cols2 = st.columns(3)
            dependents = cols2[0].selectbox("Dependents", ["Yes", "No"])
            phoneservice = cols2[1].selectbox("Phone Service", ["Yes", "No"])
            multiplelines = cols2[2].selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

            cols3 = st.columns(3)
            internet = cols3[0].selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            onlinesecurity = cols3[1].selectbox("Online Security", ["Yes", "No", "No internet service"])
            onlinebackup = cols3[2].selectbox("Online Backup", ["Yes", "No", "No internet service"])

            cols4 = st.columns(3)
            deviceprotection = cols4[0].selectbox("Device Protection", ["Yes", "No", "No internet service"])
            techsupport = cols4[1].selectbox("Tech Support", ["Yes", "No", "No internet service"])
            streamingtv = cols4[2].selectbox("Streaming TV", ["Yes", "No", "No internet service"])

            cols5 = st.columns(3)
            streamingmovies = cols5[0].selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            contract = cols5[1].selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperlessbilling = cols5[2].selectbox("Paperless Billing", ["Yes", "No"])

            paymentmethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

            tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)
            monthlycharges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
            totalcharges = st.number_input("Total Charges ($)", min_value=0.0, value=monthlycharges * tenure if tenure>0 else monthlycharges)

            submit = st.form_submit_button("Predict Churn")

        if submit:
            # prepare feature dict - keys must match model.feature_names_in_
            feature_dict = {
                'gender': gender,
                'Partner': partner,
                'Dependents': dependents,
                'PhoneService': phoneservice,
                'MultipleLines': multiplelines,
                'InternetService': internet,
                'OnlineSecurity': onlinesecurity,
                'OnlineBackup': onlinebackup,
                'DeviceProtection': deviceprotection,
                'TechSupport': techsupport,
                'StreamingTV': streamingtv,
                'StreamingMovies': streamingmovies,
                'Contract': contract,
                'PaperlessBilling': paperlessbilling,
                'PaymentMethod': paymentmethod,
                'SeniorCitizen': senior,
                'tenure': tenure,
                'MonthlyCharges': monthlycharges,
                'TotalCharges': totalcharges
            }

            # build input vector in order expected by model
            try:
                feature_names = list(model.feature_names_in_)
            except Exception:
                # fallback: try common ordering (user confirmed these features)
                feature_names = [
                    'gender','Partner','Dependents','PhoneService','MultipleLines',
                    'InternetService','OnlineSecurity','OnlineBackup','DeviceProtection',
                    'TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling',
                    'PaymentMethod','SeniorCitizen','tenure','MonthlyCharges','TotalCharges'
                ]

            input_features = []
            for feat in feature_names:
                if feat not in feature_dict:
                    st.error(f"Model expects feature '{feat}' but it's missing from the input mapping. Check model feature names.")
                    input_features = None
                    break
                val = feature_dict[feat]
                if feat in label_encoders:
                    le = label_encoders[feat]
                    enc = safe_transform_label(le, val)
                    input_features.append(enc)
                else:
                    # numeric
                    try:
                        input_features.append(float(val))
                    except Exception:
                        # fallback: try to cast booleans/strings
                        if isinstance(val, str):
                            try:
                                input_features.append(float(val))
                            except Exception:
                                input_features.append(0.0)
                        else:
                            input_features.append(0.0)

            if input_features is None:
                st.stop()

            # prediction
            try:
                pred = model.predict([input_features])[0]
                # get churn probability (probability for class=1 if available)
                prob = None
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba([input_features])[0]
                    # determine which index corresponds to churn==1
                    if hasattr(model, "classes_"):
                        classes = list(model.classes_)
                        if 1 in classes:
                            idx = classes.index(1)
                        elif "Yes" in classes:
                            idx = classes.index("Yes")
                        else:
                            # try to pick the class representing churn-like label at index 1
                            idx = 1 if len(probs) > 1 else 0
                    else:
                        idx = 1 if len(probs) > 1 else 0
                    prob = probs[idx]
                else:
                    prob = None

                churn_label = None
                if isinstance(pred, (int, np.integer)):
                    churn_label = "Yes" if int(pred) == 1 else "No"
                elif isinstance(pred, str):
                    churn_label = pred
                else:
                    churn_label = str(pred)

                if prob is not None:
                    st.markdown(f"###  Predicted churn: *{churn_label}* — Probability: *{prob*100:.2f}%*")
                else:
                    st.markdown(f"###  Predicted churn: *{churn_label}*")

                st.markdown("<div class='explain'>This prediction is made by the uploaded model using the features shown. A high probability indicates the customer is at risk of leaving — consider retention actions (discounts, outreach, bundles) for high-risk customers.</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")

# -------------------------
# Upload model/encoders
# -------------------------

# -------------------------
# Dashboard Page
# -------------------------
elif page == "Dashboard":
    st.markdown('<div class="header">Telecom Churn Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-muted">Interactive charts and plain-language explanations under each visualization.</div>', unsafe_allow_html=True)

    # try loading CSV using the uploaded file (app environment) or fallback to provided path
    DEFAULT_PATH = r"C:\Users\sravalika\Downloads\Customer Churn.csv"
    df = None

    # If user uploaded via Streamlit file_uploader earlier, allow uploading here too
    uploaded = st.file_uploader("Upload Customer Churn CSV (optional) — or leave to load default/demo", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.success("Loaded dataset from uploaded file.")
        except Exception as e:
            st.error(f"Could not read uploaded CSV: {e}")
    else:
        # try default location first
        if os.path.exists(DEFAULT_PATH):
            try:
                df = pd.read_csv(DEFAULT_PATH)
            except Exception as e:
                st.warning(f"Could not read CSV at default path: {e}")

    # fallback to demo if still None
    if df is None:
        st.info("Using a demo dataset (no CSV found). You can upload your Customer Churn.csv at the top to analyze your real data.")
        rng = np.random.default_rng(42)
        n = 700
        df = pd.DataFrame({
            "customerID": [f"CUST{i:05d}" for i in range(n)],
            "gender": rng.choice(["Male","Female"], n),
            "SeniorCitizen": rng.choice([0,1], n, p=[0.85,0.15]),
            "Partner": rng.choice(["Yes","No"], n, p=[0.45,0.55]),
            "Dependents": rng.choice(["Yes","No"], n, p=[0.28,0.72]),
            "tenure": rng.integers(0, 72, n),
            "PhoneService": rng.choice(["Yes","No"], n, p=[0.8,0.2]),
            "MultipleLines": rng.choice(["Yes","No","No phone service"], n, p=[0.35,0.45,0.2]),
            "InternetService": rng.choice(["DSL","Fiber optic","No"], n, p=[0.35,0.45,0.2]),
            "OnlineSecurity": rng.choice(["Yes","No","No internet service"], n),
            "OnlineBackup": rng.choice(["Yes","No","No internet service"], n),
            "DeviceProtection": rng.choice(["Yes","No","No internet service"], n),
            "TechSupport": rng.choice(["Yes","No","No internet service"], n),
            "StreamingTV": rng.choice(["Yes","No","No internet service"], n),
            "StreamingMovies": rng.choice(["Yes","No","No internet service"], n),
            "Contract": rng.choice(["Month-to-month","One year","Two year"], n, p=[0.55,0.25,0.2]),
            "PaperlessBilling": rng.choice(["Yes","No"], n, p=[0.6,0.4]),
            "PaymentMethod": rng.choice(["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"], n),
            "MonthlyCharges": np.round(rng.normal(70, 25, n).clip(18, 150),2)
        })
        df["TotalCharges"] = np.round(df["MonthlyCharges"] * (df["tenure"] + 1), 2)
        churn_prob = (0.5 * (df["tenure"] < 12).astype(float) + 0.5 * ((df["MonthlyCharges"] > 85).astype(float)))
        df["Churn"] = (rng.random(n) < churn_prob).astype(int).map({0:"No",1:"Yes"})

    # Ensure churn flag numeric
    if "Churn" in df.columns:
        df["Churn_flag"] = df["Churn"].map({"Yes":1,"No":0}) if df["Churn"].dtype == object else df["Churn"].astype(int)
    else:
        # try common names
        alt = [c for c in df.columns if c.lower() in ("churn","exited","is_churn")]
        if alt:
            df["Churn_flag"] = df[alt[0]].map({"Yes":1,"No":0}) if df[alt[0]].dtype==object else df[alt[0]].astype(int)
        else:
            df["Churn_flag"] = 0

    # Top KPIs
    total_customers = len(df)
    churn_count = int(df["Churn_flag"].sum())
    churn_rate = round(100 * churn_count / total_customers, 1) if total_customers else 0
    avg_tenure = round(df["tenure"].replace({np.nan:0}).mean(), 1) if "tenure" in df.columns else np.nan
    avg_monthly = round(df["MonthlyCharges"].mean(), 2) if "MonthlyCharges" in df.columns else np.nan

    k1, k2, k3, k4 = st.columns([1.4,1.1,1.1,1.1])
    with k1:
        st.markdown(f"<div class='glass kpi'><div class='label'>Total customers</div><div class='value'>{total_customers:,}</div><div class='small-muted'>Unique customers</div></div>", unsafe_allow_html=True)
    with k2:
        st.markdown(f"<div class='glass kpi'><div class='label'>Churn Rate</div><div class='value'>{churn_rate}%</div><div class='small-muted'>Customers who left</div></div>", unsafe_allow_html=True)
    with k3:
        st.markdown(f"<div class='glass kpi'><div class='label'>Avg Tenure</div><div class='value'>{avg_tenure} months</div><div class='small-muted'>Average stay length</div></div>", unsafe_allow_html=True)
    with k4:
        st.markdown(f"<div class='glass kpi'><div class='label'>Avg Monthly</div><div class='value'>${avg_monthly}</div><div class='small-muted'>Average bill</div></div>", unsafe_allow_html=True)

    st.markdown("---")

    # Row: churn by tenure + tenure distribution
    r1c1, r1c2 = st.columns([1.3,1])
    with r1c1:
        st.subheader("Churn Rate by Tenure (months)")
        if "tenure" in df.columns:
            df["tenure_bin"] = pd.cut(df["tenure"].fillna(0), bins=[-1,3,6,12,24,48,72], labels=["0-3","4-6","7-12","13-24","25-48","49+"])
            tenure_grp = df.groupby("tenure_bin").agg(total=("customerID","count" if "customerID" in df.columns else "size"),
                                                      churns=("Churn_flag","sum")).reset_index()
            tenure_grp["churn_rate"] = 100 * tenure_grp["churns"] / tenure_grp["total"]
            fig = px.bar(tenure_grp, x="tenure_bin", y="churn_rate", text=tenure_grp["churn_rate"].round(1),
                         labels={"tenure_bin":"Tenure (months)","churn_rate":"Churn Rate (%)"}, title="")
            fig.update_layout(margin=dict(t=15,b=10,l=10,r=10), height=380)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('<div class="explain"><b>What this shows:</b> Customers in their first 3 months often have the highest churn. <b>Tip:</b> prioritize onboarding and early engagement.</div>', unsafe_allow_html=True)
        else:
            st.info("No 'tenure' column found in dataset.")

    with r1c2:
        st.subheader("Tenure Distribution")
        if "tenure" in df.columns:
            fig2 = px.histogram(df, x="tenure", nbins=30, labels={"tenure":"Tenure (months)"})
            fig2.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=380)
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown('<div class="explain"><b>What this shows:</b> The distribution of customer ages. If many customers are low-tenure, churn pressure is higher.</div>', unsafe_allow_html=True)
        else:
            st.info("No 'tenure' column found in dataset.")

    st.markdown("---")

    # Middle row: service & contract analysis
    m1, m2 = st.columns(2)
    with m1:
        st.subheader("Internet Service & Contract — Churn hotspots")
        if "InternetService" in df.columns and "Contract" in df.columns:
            mech = df.groupby(["InternetService","Contract"]).agg(total=("customerID","count" if "customerID" in df.columns else "size"),
                                                                 churns=("Churn_flag","sum")).reset_index()
            mech["churn_rate"] = 100 * mech["churns"] / mech["total"]
            fig3 = px.sunburst(mech, path=["InternetService","Contract"], values="total", color="churn_rate",
                               color_continuous_scale="RdYlGn_r", hover_data=["churn_rate"])
            fig3.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=460)
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown('<div class="explain"><b>What this shows:</b> Which internet+contract combos hold the most customers and their churn. Fiber + month-to-month often appears highest. <b>Tip:</b> incentivize longer contracts for high-risk combos.</div>', unsafe_allow_html=True)
        else:
            st.info("Dataset needs 'InternetService' and 'Contract' columns for this chart.")

    with m2:
        st.subheader("Service Adoption vs Churn (Yes only)")
        service_cols = [c for c in ["PhoneService","StreamingTV","StreamingMovies","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport"] if c in df.columns]
        if service_cols:
            svc = []
            for s in service_cols:
                counts = df.groupby(s).agg(total=("customerID","count" if "customerID" in df.columns else "size"),
                                           churns=("Churn_flag","sum")).reset_index()
                counts["service"] = s
                counts["value"] = counts[s].astype(str)
                counts["churn_rate"] = 100 * counts["churns"] / counts["total"]
                svc.append(counts[ [ "service","value","total","churn_rate"] ])
            svc_df = pd.concat(svc, ignore_index=True)
            svc_yes = svc_df[svc_df["value"].str.lower()=="yes"]
            if not svc_yes.empty:
                fig4 = px.bar(svc_yes.sort_values("churn_rate", ascending=False), x="service", y="churn_rate",
                              text=svc_yes["churn_rate"].round(1), labels={"service":"Service","churn_rate":"Churn Rate (%)"})
                fig4.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=460)
                st.plotly_chart(fig4, use_container_width=True)
                st.markdown('<div class="explain"><b>What this shows:</b> For customers who have each service, we show that service\'s churn rate. Services with lower churn often indicate stronger retention value. <b>Tip:</b> push adoption of services that correlate with lower churn.</div>', unsafe_allow_html=True)
            else:
                st.info("No 'Yes' values found for service columns in your dataset.")
        else:
            st.info("No known service columns found to analyze.")

    st.markdown("---")

    # Bottom row: charges and scatter
    b1, b2 = st.columns([1.1, 1])
    with b1:
        st.subheader("Monthly Charges vs Churn (Boxplot)")
        if "MonthlyCharges" in df.columns:
            fig5 = go.Figure()
            fig5.add_trace(go.Box(y=df[df["Churn_flag"]==0]["MonthlyCharges"], name="Stayed"))
            fig5.add_trace(go.Box(y=df[df["Churn_flag"]==1]["MonthlyCharges"], name="Churned"))
            fig5.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=420, yaxis_title="Monthly Charges ($)")
            st.plotly_chart(fig5, use_container_width=True)
            st.markdown('<div class="explain"><b>What this shows:</b> Compare billing distributions of those who stayed vs churned. If churners have higher bills, pricing may be a driver. <b>Tip:</b> consider targeted offers for high-bill at-risk customers.</div>', unsafe_allow_html=True)
        else:
            st.info("No 'MonthlyCharges' column found.")

    with b2:
        st.subheader("Customer Segments: Tenure vs MonthlyCharges")
        if "tenure" in df.columns and "MonthlyCharges" in df.columns:
            sample_df = df.sample(min(len(df), 2000))
            fig6 = px.scatter(sample_df, x="tenure", y="MonthlyCharges",
                              color=sample_df["Churn_flag"].map({0:"Stayed",1:"Churned"}),
                              labels={"tenure":"Tenure (months)", "MonthlyCharges":"Monthly Charges ($)"},
                              hover_data=["customerID"] if "customerID" in df.columns else None)
            fig6.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=420)
            st.plotly_chart(fig6, use_container_width=True)
            st.markdown('<div class="explain"><b>What this shows:</b> Clusters of customers by tenure and bill amount. Look for the high-bill, low-tenure quadrant — top candidates for retention campaigns.</div>', unsafe_allow_html=True)
        else:
            st.info("Need both 'tenure' and 'MonthlyCharges' columns for this chart.")

    st.markdown("---")
    #st.subheader("Plain-language Recommendations")
    ##- *Reduce early churn (0–3 months):* focus onboarding: welcome calls, quick support, small incentives.  
    #- *Target high-bill, short-tenure customers:* offer bundles or small loyalty discounts early.  
    #- *Encourage longer contracts for risky segments:* push One-year / Two-year with perks.  
    #- *Promote value services:* tech support and device protection often reduce churn.  
    #- *Monitor KPIs weekly* to measure improvements.
   # """, unsafe_allow_html=True)

   ## st.dataframe(df.head(8))

# -------------------------
# About Page
# -------------------------
