import streamlit as st
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(page_title="Customer Segmentation", layout="wide")

# Enhanced custom styling with typewriter and hover effects
st.markdown("""
<style>
* {
    margin: 0;
    padding: 0;
}

body {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Typewriter Effect */
@keyframes typewriter {
    0% { width: 0; }
    100% { width: 100%; }
}

@keyframes blink {
    50% { border-right-color: transparent; }
}

.typewriter {
    overflow: hidden;
    border-right: 3px solid #00C897;
    white-space: nowrap;
    animation: typewriter 4s steps(60, end), blink 0.5s step-end infinite;
    font-size: 2.5em;
    font-weight: bold;
    color: #00C897;
}

/* Card Hover Effects */
.card {
    padding: 20px;
    border-radius: 15px;
    background: linear-gradient(135deg, #ffffff 0%, #f5f5f5 100%);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
    cursor: pointer;
}

.card:hover {
    transform: translateY(-10px) scale(1.02);
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    background: linear-gradient(135deg, #f5f5f5 0%, #ffffff 100%);
}

.metric-card {
    padding: 25px;
    border-radius: 12px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
    transition: all 0.3s ease;
    border-left: 5px solid #00C897;
}

.metric-card:hover {
    transform: translateX(5px);
    box-shadow: 0 6px 20px rgba(0, 200, 151, 0.3);
}

.cluster-info {
    padding: 15px;
    border-radius: 10px;
    background-color: #e8f5e9;
    margin: 10px 0;
    border-left: 5px solid #00C897;
    transition: all 0.3s ease;
}

.cluster-info:hover {
    background-color: #c8e6c9;
    transform: translateX(5px);
}

/* Button Styling */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 30px;
    font-weight: bold;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}

.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 6px 25px rgba(102, 126, 234, 0.5);
}

/* Result Boxes */
.success-box {
    padding: 20px;
    border-radius: 12px;
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    color: white;
    box-shadow: 0 4px 15px rgba(17, 153, 142, 0.3);
    transition: all 0.3s ease;
}

.success-box:hover {
    transform: scale(1.02);
    box-shadow: 0 6px 20px rgba(17, 153, 142, 0.5);
}

.warning-box {
    padding: 20px;
    border-radius: 12px;
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
    box-shadow: 0 4px 15px rgba(245, 87, 108, 0.3);
    transition: all 0.3s ease;
}

.warning-box:hover {
    transform: scale(1.02);
    box-shadow: 0 6px 20px rgba(245, 87, 108, 0.5);
}

/* Section Headers */
h2 {
    margin-top: 30px;
    margin-bottom: 20px;
    color: #333;
    border-bottom: 3px solid #667eea;
    padding-bottom: 10px;
}

/* Fade In Animation */
@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.fade-in {
    animation: fadeIn 0.8s ease-in-out;
}
</style>
""", unsafe_allow_html=True)

# Typewriter Title
st.markdown("""
<div class="typewriter">
🛒 Customer Segmentation & Prediction
</div>
<p style='text-align: center; color: #666; font-size: 1.1em; margin-top: 10px;'>
Explore customer segments and predict purchase behavior with AI 🤖
</p>
""", unsafe_allow_html=True)

st.markdown("<hr style='border: 2px solid #667eea;'>", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        model = joblib.load("Models/classifier.pkl")
        kmeans = joblib.load("Models/kmeans.pkl")
        scaler = joblib.load("Models/scaler.pkl")
        return model, kmeans, scaler
    except FileNotFoundError as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

model, kmeans, scaler = load_models()

if model is None or kmeans is None or scaler is None:
    st.stop()

# Cluster metadata for visualizations
cluster_colors = {
    0: "#00C897",
    1: "#FF6B6B",
    2: "#FFD93D",
    3: "#6C5CE7"
}

cluster_names = {
    0: "🟢 Regular Customers",
    1: "🔴 At Risk Customers",
    2: "💰 High Value Customers",
    3: "👑 Premium Customers"
}

cluster_info_dict = {
    0: {"emoji": "✅", "color": "success", "desc": "Consistent, recent purchaser"},
    1: {"emoji": "⚠️", "color": "warning", "desc": "Hasn't purchased recently"},
    2: {"emoji": "🌟", "color": "success", "desc": "Extremely valuable"},
    3: {"emoji": "👑", "color": "success", "desc": "Ultra-high value"}
}

# Sidebar
st.sidebar.header("📊 Customer Details")

# Cluster guide with better styling
with st.sidebar.expander("📋 Cluster Guide", expanded=True):
    st.markdown("""
    ### 🟢 Cluster 0 - Active Regular
    - **Recency:** 40 days
    - **Frequency:** 105 purchases
    - **Spend:** $2,000
    
    ---
    
    ### 🔴 Cluster 1 - At-Risk
    - **Recency:** 250 days
    - **Frequency:** 28 purchases
    - **Spend:** $465
    
    ---
    
    ### 🟡 Cluster 2 - High Value Customers
    - **Recency:** 2 days
    - **Frequency:** 4,800 purchases
    - **Spend:** $55,000
    
    ---
    
    ### 🟣 Cluster 3 - Premium Customers
    - **Recency:** 9 days
    - **Frequency:** 1,000 purchases
    - **Spend:** $192,000
    """)

# Input sliders with better styling
st.sidebar.markdown("### 🎯 Adjust Customer Profile")
recency = st.sidebar.slider("📅 Recency (days since last purchase)", 0, 365, 40, step=5)
frequency = st.sidebar.slider("🛍️ Frequency (total purchases)", 1, 5000, 105, step=10)
monetary = st.sidebar.slider("💰 Monetary (total spend $)", 10, 200000, 2000, step=100)

# Prediction
if st.sidebar.button("🔮 Predict", use_container_width=True):
    
    # Display input metrics with better styling
    st.markdown("## 📋 Input Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
        <h4>📅 Recency</h4>
        <p style="font-size: 2em; margin: 10px 0;">{recency} days</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
        <h4>🛍️ Frequency</h4>
        <p style="font-size: 2em; margin: 10px 0;">{frequency}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
        <h4>💰 Monetary</h4>
        <p style="font-size: 2em; margin: 10px 0;">${monetary:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Make predictions
    input_data = np.array([[recency, frequency, monetary]])
    scaled_data = scaler.transform(input_data)
    
    cluster = kmeans.predict(scaled_data)[0]
    prediction = model.predict(input_data)[0]
    
    # Display results with enhanced styling
    st.markdown("## 🎯 Prediction Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cluster_name = cluster_names.get(cluster, f'Cluster {cluster}')
        cluster_color = cluster_colors.get(cluster, '#667eea')
        st.markdown(f"""
        <div style="
            padding: 25px;
            border-radius: 12px;
            background: {cluster_color};
            color: white;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            border: 3px solid white;
        ">
        <h3 style="margin: 0 0 10px 0;">🎯 CLUSTER</h3>
        <p style="font-size: 2em; font-weight: bold; margin: 0;">{cluster}</p>
        <p style="font-size: 1.2em; margin: 10px 0 0 0;">{cluster_name}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if prediction == 1:
            value_color = "#11998e"
            value_text = "HIGH VALUE"
            value_emoji = "💰"
        else:
            value_color = "#f5576c"
            value_text = "LOW VALUE"
            value_emoji = "⚠️"
        
        st.markdown(f"""
        <div style="
            padding: 25px;
            border-radius: 12px;
            background: {value_color};
            color: white;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            border: 3px solid white;
        ">
        <h3 style="margin: 0 0 10px 0;">{value_emoji} VALUE</h3>
        <p style="font-size: 2em; font-weight: bold; margin: 0;">{value_text}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        risk_level = 'Low' if recency < 100 else 'Medium' if recency < 250 else 'High'
        risk_emoji = '🟢' if recency < 100 else '🟡' if recency < 250 else '🔴'
        risk_color = '#38ef7d' if recency < 100 else '#FFD93D' if recency < 250 else '#FF6B6B'
        
        st.markdown(f"""
        <div style="
            padding: 25px;
            border-radius: 12px;
            background: {risk_color};
            color: white;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            border: 3px solid white;
        ">
        <h3 style="margin: 0 0 10px 0;">⭐ RISK</h3>
        <p style="font-size: 2em; font-weight: bold; margin: 0;">{risk_emoji}</p>
        <p style="font-size: 1.2em; margin: 10px 0 0 0;">{risk_level}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Key Metrics in a clean layout
    st.markdown("## 📊 Customer Details")
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("📅 Recency", f"{recency}d", "Days since purchase")
    
    with metric_col2:
        st.metric("🛍️ Frequency", f"{frequency}", "Total purchases")
    
    with metric_col3:
        st.metric("💵 Monetary", f"${monetary:,.0f}", "Total spend")
    
    with metric_col4:
        cluster_benchmarks = {
            0: {"Recency": 40, "Frequency": 105, "Monetary": 1994},
            1: {"Recency": 246, "Frequency": 28, "Monetary": 465},
            2: {"Recency": 2, "Frequency": 4821, "Monetary": 55039},
            3: {"Recency": 9, "Frequency": 1013, "Monetary": 192103}
        }
        benchmark = cluster_benchmarks.get(cluster, {})
        avg_monetary = benchmark.get("Monetary", 0)
        diff = ((monetary - avg_monetary) / avg_monetary * 100) if avg_monetary > 0 else 0
        delta_text = f"+{diff:.0f}% vs avg" if diff > 0 else f"{diff:.0f}% vs avg"
        st.metric("📈 vs Cluster", f"{abs(diff):.0f}%", delta_text)
    
    # Cluster Information Box
    st.markdown("---")
    
    cluster_descriptions = {
        0: {
            "title": "✅ Active Regular Customer",
            "bg": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            "desc": "Consistently engaged customers with steady purchase behavior. High potential to increase lifetime value through upselling and deeper engagement.",
                "recommendations":[
                "🎯 Personalized product recommendations",
                "🔄 Upsell & cross-sell mid/high-value items",
                "🎁 Loyalty points & milestone rewards",
                "📩 Regular engagement via email/app notifications"
            ]
        },
        1: {
            "title": "⚠️ At-Risk Customer",
            "bg": "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
            "desc": "Previously active customers showing declining engagement. High churn risk — immediate action required to retain them.",
            "recommendations": [
                "🚀 Win-back campaigns with urgency (limited-time offers)",
                "💰 Strong discounts or cashback incentives",
                "📊 Feedback surveys to identify drop-off reasons",
                "🎯 Retargeting ads & personalized reactivation messages"
            ]
        },
        2: {
            "title": "🌟 High Value Customers",
            "bg": "linear-gradient(135deg, #FFD93D 0%, #FF6B6B 100%)",
            "desc": "Top revenue contributors with high purchase frequency and value. Critical for profitability and brand advocacy.",
            "recommendations": [
                "👑 Exclusive VIP perks & early access",
                "🎁 Premium rewards & surprise gifts",
                "🤝 Dedicated support / priority service",
                "📣 Encourage referrals & brand advocacy"
            ]
        },
        3: {
            "title": "👑 Premium Customer",
            "bg": "linear-gradient(135deg, #6C5CE7 0%, #A29BFE 100%)",
            "desc": "Elite segment with extremely high value and strategic importance. Long-term relationship building is key.",
            "recommendations": [
                "💼 Personalized experiences & custom offerings",
                "🏆 White-glove / concierge-level service",
                "🤝 Co-creation opportunities (feedback → product input)",
                "🎯 Priority support, faster delivery, exclusive deals"
            ]
        }
    }
    
    cluster_info = cluster_descriptions.get(cluster, {})
    
    st.markdown(f"""
    <div style="
        padding: 25px;
        border-radius: 15px;
        background: {cluster_info['bg']};
        color: white;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
    ">
    <h2 style="margin-top: 0; color: white;">{cluster_info['title']}</h2>
    <p style="font-size: 1.1em; margin: 15px 0;">{cluster_info['desc']}</p>
    
    <h4 style="color: white;">📋 Recommended Actions:</h4>
    <ul style="margin: 10px 0; padding-left: 20px;">
    """, unsafe_allow_html=True)
    
    for rec in cluster_info['recommendations']:
        st.markdown(f"<li style='margin: 8px 0; font-size: 1em;'>{rec}</li>", unsafe_allow_html=True)
    
    st.markdown("</ul></div>", unsafe_allow_html=True)

st.divider()

# Footer
st.markdown("""
<p style='text-align: center; color: #999; font-size: 12px; margin-top: 30px;'>
✨ Customer Segmentation Dashboard | Powered by ML Models & Streamlit ✨
</p>
""", unsafe_allow_html=True)
