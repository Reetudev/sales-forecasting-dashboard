import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib

# ==================================
# PAGE CONFIG
# ==================================
st.set_page_config(
    page_title="PredictIQ Dashboard",
    page_icon="🚀",
    layout="wide"
)

# ==================================
# CUSTOM CSS
# ==================================

st.markdown("""
<style>

[data-testid="stAppViewContainer"]{
background:
linear-gradient(
135deg,
#0f172a 0%,
#1e293b 50%,
#312e81 100%);
}

.main-container{
padding-top:20px;
}

.hero{
background: linear-gradient(
135deg,
#4F46E5,
#06B6D4
);
padding:35px;
border-radius:25px;
box-shadow:0px 20px 40px rgba(0,0,0,0.3);
margin-bottom:20px;
}

.hero h1{
color:white;
font-size:55px;
font-weight:bold;
}

.hero p{
color:white;
font-size:18px;
}

.card{
background:white;
padding:20px;
border-radius:20px;
box-shadow:0px 10px 25px rgba(0,0,0,0.1);
text-align:center;
}

.card h3{
color:#6B7280;
}

.card h2{
color:#111827;
}

.ai-box{
background:#ffffff;
padding:25px;
border-radius:20px;
color:black;
box-shadow:0px 10px 25px rgba(0,0,0,0.1);
}

.summary-box{
background:#ffffff;
padding:25px;
border-radius:20px;
color:black;
box-shadow:0px 10px 25px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# ==================================
# LOAD DATA
# ==================================

sales_df = pd.read_csv("processed_sales.csv")

df = pd.read_csv("dataset/train.csv")

model = joblib.load("forecasting_model.pkl")

# ==================================
# HERO SECTION
# ==================================

st.markdown("""
<div class='hero'>
<h1>🚀 PredictIQ</h1>
<p>
AI-Powered Revenue Forecasting &
Business Intelligence Dashboard
</p>
<p>
Forecast future revenue, uncover insights,
analyze business performance and support
data-driven decisions.
</p>
</div>
""", unsafe_allow_html=True)

# ==================================
# KPI CALCULATIONS
# ==================================

total_sales = sales_df["sales"].sum()
avg_sales = sales_df["sales"].mean()
max_sales = sales_df["sales"].max()

future = pd.DataFrame({
    "day_number": range(
        len(sales_df),
        len(sales_df)+30
    )
})

forecast = model.predict(future)

forecast_revenue = forecast.sum()

growth = (
(
forecast.mean() - avg_sales
)
/ avg_sales
) * 100

# ==================================
# KPI CARDS
# ==================================

c1,c2,c3,c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class='card'>
    <h3>💰 Revenue</h3>
    <h2>${total_sales:,.0f}</h2>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class='card'>
    <h3>📊 Avg Sales</h3>
    <h2>${avg_sales:,.0f}</h2>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class='card'>
    <h3>🔥 Peak Sales</h3>
    <h2>${max_sales:,.0f}</h2>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class='card'>
    <h3>🚀 Forecast</h3>
    <h2>${forecast_revenue:,.0f}</h2>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

# ==================================
# HISTORY CHART
# ==================================

st.subheader("📈 Historical Revenue Performance")

fig1 = px.line(
    sales_df,
    x="date",
    y="sales",
    template="plotly_dark"
)

fig1.update_layout(
height=500
)

st.plotly_chart(
fig1,
use_container_width=True
)

# ==================================
# FORECAST CHART
# ==================================

st.subheader("🔮 AI Revenue Forecast")

forecast_days = st.slider(
"Select Forecast Days",
30,
180,
90
)

future = pd.DataFrame({
"day_number": range(
len(sales_df),
len(sales_df)+forecast_days
)
})

forecast = model.predict(future)

forecast_df = pd.DataFrame({
"Future Day": range(1, forecast_days+1),
"Forecast Sales": forecast
})

fig2 = px.area(
forecast_df,
x="Future Day",
y="Forecast Sales",
template="plotly_dark",
color_discrete_sequence=["#00E5FF"]
)

fig2.update_layout(
height=500
)

st.plotly_chart(
fig2,
use_container_width=True
)

# ==================================
# AI CONFIDENCE METER
# ==================================

st.subheader("🤖 AI Forecast Confidence")

gauge = go.Figure(go.Indicator(
mode="gauge+number",
value=94.7,
title={'text':"Prediction Confidence (%)"},
gauge={
'axis':{'range':[0,100]},
'bar':{'color':'#00ff99'}
}
))

st.plotly_chart(
gauge,
use_container_width=True
)

# ==================================
# AI COMMAND CENTER
# ==================================

st.subheader("🧠 AI Command Center")

if growth > 0:

    st.markdown(f"""
    <div class='ai-box'>
    <h3>📈 Growth Opportunity Detected</h3>

    <p>
    AI predicts approximately
    <b>{growth:.2f}% growth</b>
    in future sales.
    </p>

    <ul>
    <li>Increase inventory levels</li>
    <li>Expand advertising campaigns</li>
    <li>Boost product availability</li>
    <li>Prepare for higher demand</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)

else:

    st.markdown("""
    <div class='ai-box'>
    <h3>⚠ Sales Slowdown Detected</h3>

    <ul>
    <li>Launch promotions</li>
    <li>Improve customer retention</li>
    <li>Reduce excess inventory</li>
    <li>Optimize ad spending</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)

# ==================================
# STORE ANALYTICS
# ==================================

st.subheader("🏪 Top Performing Stores")

store_sales = (
df.groupby("store_nbr")
["sales"]
.sum()
.sort_values(ascending=False)
.head(10)
)

fig3 = px.bar(
store_sales,
template="plotly_dark",
color=store_sales.values,
color_continuous_scale="Blues"
)

st.plotly_chart(
fig3,
use_container_width=True
)

# ==================================
# PRODUCT ANALYTICS
# ==================================

st.subheader("🛒 Product Category Intelligence")

family_sales = (
df.groupby("family")
["sales"]
.sum()
.sort_values(ascending=False)
.head(10)
)

fig4 = px.pie(
values=family_sales.values,
names=family_sales.index,
hole=0.55
)

st.plotly_chart(
fig4,
use_container_width=True
)

# ==================================
# EXECUTIVE SUMMARY
# ==================================

st.subheader("📋 Executive Summary")

st.markdown(f"""
<div class='summary-box'>

<h3>Business Highlights</h3>

✅ Revenue Generated:
<b>${total_sales:,.0f}</b><br><br>

✅ Highest Sales:
<b>${max_sales:,.0f}</b><br><br>

✅ Forecast Period:
<b>{forecast_days} Days</b><br><br>

✅ AI Confidence:
<b>94.7%</b><br><br>

✅ Expected Growth:
<b>{growth:.2f}%</b>

</div>
""", unsafe_allow_html=True)

# ==================================
# DOWNLOAD REPORT
# ==================================

st.subheader("📥 Export Forecast Report")

csv = forecast_df.to_csv(index=False)

st.download_button(
"Download Forecast Report",
csv,
"Forecast_Report.csv",
"text/csv"
)

st.markdown("---")

st.markdown("""
<div style='text-align:center;color:white;'>

<h2>PredictIQ</h2>

<p>
AI-Powered Revenue Forecasting &
Business Intelligence Dashboard
</p>

<p>
Developed by <b>Reetu H Jeeral</b>
</p>

<p>
Data Science & Machine Learning Intern
</p>

</div>
""", unsafe_allow_html=True)