# ============================================================
# STREAMLIT APP: MODERN CARD-BASED DASHBOARD
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np

# -------------------------------
# APP CONFIG
# -------------------------------
st.set_page_config(
    page_title="User Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        padding-bottom: 1rem;
        border-bottom: 3px solid #3B82F6;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 20px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #F3F4F6;
        border-radius: 5px 5px 0px 0px;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<h1 class="main-header">👥 User Intelligence Dashboard</h1>', unsafe_allow_html=True)
with col2:
    st.markdown("### 📅 Last Updated: Today")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/app_user_behavior_dataset.csv")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

df = load_data()

# -------------------------------
# TOP METRICS ROW
# -------------------------------
st.markdown("## 📈 Key Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h3>👥 Total Users</h3>
        <h2>50,000</h2>
        <p>Active this month</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    avg_engagement = df['sessions_per_week'].mean()
    st.markdown(f"""
    <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
        <h3>📊 Avg Sessions</h3>
        <h2>{avg_engagement:.1f}</h2>
        <p>Per user per week</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    churn_rate = (df['churn_risk_score'] > 0.7).mean() * 100
    st.markdown(f"""
    <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
        <h3>⚠️ Churn Risk</h3>
        <h2>{churn_rate:.1f}%</h2>
        <p>High risk users</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    avg_duration = df['avg_session_duration_min'].mean()
    st.markdown(f"""
    <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
        <h3>⏱️ Avg Session</h3>
        <h2>{avg_duration:.1f}m</h2>
        <p>Duration per session</p>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# TABS FOR DIFFERENT VIEWS
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Clustering", "📊 Analytics", "👤 User Profile", "📈 Trends"])

with tab1:
    # Clustering controls in sidebar
    st.sidebar.header("🎛️ Clustering Settings")
    k = st.sidebar.slider("Number of Clusters", 2, 8, 4)
    selected_features = st.sidebar.multiselect(
        "Select Features for Clustering",
        df.select_dtypes(include=[np.number]).columns.tolist(),
        default=['sessions_per_week', 'avg_session_duration_min', 'daily_active_minutes']
    )
    
    # Perform clustering
    if selected_features:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[selected_features])
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Cluster distribution - Pie chart
        cluster_counts = df['cluster'].value_counts().sort_index()
        fig = px.pie(
            values=cluster_counts.values,
            names=[f'Cluster {i}' for i in cluster_counts.index],
            title='📊 Cluster Distribution',
            hole=0.3,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Parallel coordinates plot
        fig = px.parallel_categories(
            df,
            dimensions=['cluster', 'gender', 'device_type'],
            title='🔗 Cluster Demographics',
            color_continuous_scale=px.colors.sequential.Plasma
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Cluster profiles
    st.markdown("### 📋 Cluster Characteristics")
    cluster_profile = df.groupby('cluster')[selected_features].mean().round(2)
    st.dataframe(cluster_profile.style.background_gradient(cmap='Blues'))

with tab2:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Heatmap of correlations
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:8]
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            title='🔗 Feature Correlation Matrix',
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Distribution by Device")
        device_counts = df['device_type'].value_counts()
        fig = px.bar(
            x=device_counts.index,
            y=device_counts.values,
            title='📱 Device Usage',
            color=device_counts.index,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Histograms for key metrics
    st.markdown("### 📈 Feature Distributions")
    selected_metric = st.selectbox("Select Metric", numeric_cols)
    fig = px.histogram(
        df,
        x=selected_metric,
        nbins=50,
        title=f'Distribution of {selected_metric}',
        marginal='box',
        color_discrete_sequence=['#636EFA']
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### 👤 User Profile Explorer")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # User selection
        user_id = st.selectbox("Select User ID", df['user_id'].head(100).tolist())
        user_data = df[df['user_id'] == user_id].iloc[0]
    
    with col2:
        st.metric("Age", user_data['age'])
        st.metric("Gender", user_data['gender'])
        st.metric("Country", user_data['country'])
    
    # Radar chart for user profile
    metrics = ['sessions_per_week', 'avg_session_duration_min', 
               'daily_active_minutes', 'feature_clicks_per_session', 
               'pages_viewed_per_session']
    
    user_values = [user_data[m] for m in metrics]
    avg_values = [df[m].mean() for m in metrics]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=user_values,
        theta=metrics,
        fill='toself',
        name='Selected User',
        line_color='blue'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=avg_values,
        theta=metrics,
        fill='toself',
        name='Average User',
        line_color='red'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(user_values), max(avg_values)) * 1.2]
            )),
        showlegend=True,
        title="📊 User vs Average Profile"
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    # Time series simulation (if you have date data)
    st.markdown("### 📅 Engagement Trends")
    
    # Simulate trend data
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    trend_data = pd.DataFrame({
        'date': dates,
        'active_users': np.random.randint(2000, 5000, 30),
        'avg_session_min': np.random.uniform(10, 30, 30),
        'churn_rate': np.random.uniform(5, 20, 30)
    })
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('📈 Daily Active Users', '⏱️ Average Session Duration')
    )
    
    fig.add_trace(
        go.Scatter(x=trend_data['date'], y=trend_data['active_users'],
                  mode='lines+markers', name='Active Users',
                  line=dict(color='#636EFA')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=trend_data['date'], y=trend_data['avg_session_min'],
                  mode='lines+markers', name='Session Duration',
                  line=dict(color='#EF553B')),
        row=2, col=1
    )
    
    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# SIDEBAR ADDITIONAL CONTROLS
# -------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Filter Options")
selected_country = st.sidebar.multiselect(
    "Select Countries",
    df['country'].unique().tolist(),
    default=df['country'].unique()[:3]
)

selected_device = st.sidebar.multiselect(
    "Select Devices",
    df['device_type'].unique().tolist(),
    default=df['device_type'].unique()
)

age_range = st.sidebar.slider(
    "Age Range",
    min_value=int(df['age'].min()),
    max_value=int(df['age'].max()),
    value=(20, 50)
)

# -------------------------------
# DOWNLOAD SECTION
# -------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 💾 Export Data")
if st.sidebar.button("📥 Download Report"):
    st.sidebar.success("Report generated! (Simulated)")