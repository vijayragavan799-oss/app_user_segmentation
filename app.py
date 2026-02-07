import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image


# ----------------------------
# Page Config
# ----------------------------

st.set_page_config(
    page_title="App User Segmentation Dashboard",
    layout="wide"
)


# ----------------------------
# Title
# ----------------------------

st.title("📊 App User Segmentation Dashboard")
st.markdown("Unsupervised Machine Learning based User Analysis")


# ----------------------------
# Load Data
# ----------------------------

@st.cache_data
def load_data():
    return pd.read_csv("outputs/clustered_users.csv")


df = load_data()


# ----------------------------
# Sidebar Navigation
# ----------------------------

st.sidebar.header("Navigation")

option = st.sidebar.selectbox(
    "Select View",
    ["Overview", "Cluster Analysis", "User Explorer", "Visualizations"],
    key="main_navigation"
)


# ----------------------------
# OVERVIEW
# ----------------------------

if option == "Overview":

    st.subheader("📌 Project Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Users", len(df))
    col2.metric("Clusters", df["Cluster"].nunique())
    col3.metric("At-Risk Users", len(df[df["Cluster"] == 2]))
    col4.metric("Countries", df["country"].nunique())

    st.markdown("---")

    st.subheader("Cluster Distribution")

    fig = px.pie(
        df,
        names="Cluster",
        title="User Distribution by Cluster"
    )

    st.plotly_chart(fig, use_container_width=True)


# ----------------------------
# CLUSTER ANALYSIS
# ----------------------------

elif option == "Cluster Analysis":

    st.subheader("📈 Cluster Performance Analysis")

    # Engagement
    fig1 = px.box(
        df,
        x="Cluster",
        y="engagement_score",
        title="Engagement Score by Cluster"
    )

    st.plotly_chart(fig1, use_container_width=True)

    # Churn Risk
    fig2 = px.box(
        df,
        x="Cluster",
        y="churn_risk_score",
        title="Churn Risk by Cluster"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # Session Duration
    avg_session = df.groupby("Cluster")["avg_session_duration_min"].mean().reset_index()

    fig3 = px.bar(
        avg_session,
        x="Cluster",
        y="avg_session_duration_min",
        title="Average Session Duration per Cluster"
    )

    st.plotly_chart(fig3, use_container_width=True)


# ----------------------------
# USER EXPLORER (TABLE VIEW)
# ----------------------------

elif option == "User Explorer":

    st.subheader("🔍 Explore Users by Cluster (Table View)")

    # Select Cluster
    selected_cluster = st.selectbox(
        "Select Cluster",
        sorted(df["Cluster"].unique()),
        key="user_explorer_cluster"
    )

    filtered_df = df[df["Cluster"] == selected_cluster]

    # -------------------------
    # KPIs
    # -------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric("Users", len(filtered_df))
    col2.metric(
        "Avg Engagement",
        round(filtered_df["engagement_score"].mean(), 2)
    )
    col3.metric(
        "Avg Churn Risk",
        round(filtered_df["churn_risk_score"].mean(), 2)
    )

    st.markdown("---")

    # -------------------------
    # User Table
    # -------------------------

    st.subheader("📄 Users in Selected Cluster")

    st.dataframe(filtered_df, use_container_width=True)

    st.markdown("---")

    # -------------------------
    # Device Summary
    # -------------------------

    st.subheader("📱 Device Distribution")

    device_table = (
        filtered_df["device_type"]
        .value_counts()
        .reset_index()
    )

    device_table.columns = ["Device Type", "Users"]

    st.dataframe(device_table, use_container_width=True)

    # -------------------------
    # Country Summary
    # -------------------------

    st.subheader("🌍 Country Distribution")

    country_table = (
        filtered_df["country"]
        .value_counts()
        .reset_index()
    )

    country_table.columns = ["Country", "Users"]

    st.dataframe(country_table, use_container_width=True)

    # -------------------------
    # Engagement Summary
    # -------------------------

    st.subheader("📊 Engagement Summary")

    engagement_summary = filtered_df[
        ["engagement_score", "sessions_per_week", "avg_session_duration_min"]
    ].describe().round(2)

    st.dataframe(engagement_summary)

    st.markdown("---")

    # -------------------------
    # Download Button
    # -------------------------

    st.download_button(
        "⬇️ Download This Cluster Data",
        filtered_df.to_csv(index=False),
        file_name=f"cluster_{selected_cluster}_users.csv"
    )


# ----------------------------
# VISUALIZATIONS
# ----------------------------

elif option == "Visualizations":

    st.subheader("📉 PCA Visualization")

    pca_img = Image.open("outputs/plots/pca_clusters.png")
    st.image(pca_img, use_column_width=True)

    st.subheader("📈 Elbow Method")

    elbow_img = Image.open("outputs/plots/elbow_method.png")
    st.image(elbow_img, use_column_width=True)

    st.markdown("---")

    st.subheader("Engagement vs Sessions")

    fig = px.scatter(
        df,
        x="sessions_per_week",
        y="engagement_score",
        color="Cluster",
        title="Engagement vs Sessions"
    )

    st.plotly_chart(fig, use_container_width=True)
