import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# For scaling and clustering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# SOM
try:
    from minisom import MiniSom
    MINISOM_AVAILABLE = True
except ImportError:
    MINISOM_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .cluster-name {
        font-weight: bold;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading and processing
@st.cache_data
def load_and_process_data():
    """Load and process the customer data"""
    try:
        file_path = "datasets/online_retail.csv"    
        data = pd.read_csv(file_path)
        
        # Remove rows without customer ID
        data = data.dropna(subset=["Customer ID"])
        
        # Convert InvoiceDate to datetime
        data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])
        
        # Identify cancellations (Invoice starting with 'C')
        data["IsCancelled"] = data["Invoice"].astype(str).str.startswith("C")
        
        # Create TotalAmount column
        data["TotalAmount"] = data["Quantity"] * data["Price"]
        
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def create_rfmc_features(data):
    """Create RFMC features from the data"""
    ref_date = data["InvoiceDate"].max() + pd.Timedelta(days=1)
    
    rfmc = data.groupby("Customer ID").agg({
        "InvoiceDate": lambda x: (ref_date - x.max()).days,
        "Invoice": "nunique",
        "TotalAmount": "sum",
        "IsCancelled": "mean"
    }).reset_index()
    
    rfmc.columns = ["CustomerID", "Recency", "Frequency", "Monetary", "CancellationRate"]
    return rfmc

@st.cache_data
def perform_clustering(rfmc):
    """Perform clustering using multiple algorithms"""
    features = ["Recency", "Frequency", "Monetary", "CancellationRate"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfmc[features])
    
    results = {}
    
    def evaluate_clusters(X, labels):
        return {
            "silhouette": silhouette_score(X, labels),
            "davies_bouldin": davies_bouldin_score(X, labels),
            "calinski_harabasz": calinski_harabasz_score(X, labels)
        }
    
    # KMeans
    kmeans = KMeans(n_clusters=4, random_state=42)
    rfmc_copy = rfmc.copy()
    rfmc_copy["Cluster_KMeans"] = kmeans.fit_predict(X_scaled)
    results["KMeans"] = evaluate_clusters(X_scaled, rfmc_copy["Cluster_KMeans"])
    
    # Hierarchical
    hier = AgglomerativeClustering(n_clusters=4)
    rfmc_copy["Cluster_Hier"] = hier.fit_predict(X_scaled)
    results["Hierarchical"] = evaluate_clusters(X_scaled, rfmc_copy["Cluster_Hier"])
    
    # GMM
    gmm = GaussianMixture(n_components=4, random_state=42)
    rfmc_copy["Cluster_GMM"] = gmm.fit_predict(X_scaled)
    results["GMM"] = evaluate_clusters(X_scaled, rfmc_copy["Cluster_GMM"])
    
    # SOM (if available)
    if MINISOM_AVAILABLE:
        som = MiniSom(x=2, y=2, input_len=4, sigma=1.0, learning_rate=0.5, random_seed=42)
        som.random_weights_init(X_scaled)
        som.train_random(X_scaled, 100)
        
        som_clusters = []
        for row in X_scaled:
            winner = som.winner(row)
            som_clusters.append(winner[0] * 2 + winner[1])
        rfmc_copy["Cluster_SOM"] = som_clusters
        results["SOM"] = evaluate_clusters(X_scaled, rfmc_copy["Cluster_SOM"])
    
    # Add cluster names
    cluster_name_map = {
        0: "Loyal Regulars",
        1: "At Risk",
        2: "Inactive", 
        3: "Champions"
    }
    
    for method in ["KMeans", "Hier", "GMM"] + (["SOM"] if MINISOM_AVAILABLE else []):
        rfmc_copy[f"Cluster_{method}_Name"] = rfmc_copy[f"Cluster_{method}"].map(cluster_name_map)
    
    return rfmc_copy, results, X_scaled, scaler

def main():
    st.markdown('<h1 class="main-header">🎯 Customer Segmentation Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading data..."):
        data = load_and_process_data()
        
    if data is None:
        st.error("Failed to load data. Please check the data source.")
        return
        
    rfmc = create_rfmc_features(data)
    rfmc_clustered, results, X_scaled, scaler = perform_clustering(rfmc)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["📊 Overview", "🔍 Cluster Analysis", "👤 Individual Customer Analysis", "📈 Model Performance"]
    )
    
    if page == "📊 Overview":
        show_overview(rfmc_clustered, data)
    elif page == "🔍 Cluster Analysis":
        show_cluster_analysis(rfmc_clustered)
    elif page == "👤 Individual Customer Analysis":
        show_individual_analysis(rfmc_clustered, data)
    elif page == "📈 Model Performance":
        show_model_performance(results)

def show_overview(rfmc_clustered, data):
    st.header("📊 Business Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", len(rfmc_clustered))
    with col2:
        st.metric("Total Revenue", f"£{data['TotalAmount'].sum():,.2f}")
    with col3:
        st.metric("Total Orders", data['Invoice'].nunique())
    with col4:
        st.metric("Avg Order Value", f"£{data['TotalAmount'].mean():.2f}")
    
    # Customer distribution by cluster
    st.subheader("Customer Distribution by Segment (K-Means)")
    
    cluster_dist = rfmc_clustered['Cluster_KMeans_Name'].value_counts()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        fig = px.pie(
            values=cluster_dist.values,
            names=cluster_dist.index,
            title="Customer Distribution by Segment"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Revenue by cluster
        revenue_by_cluster = rfmc_clustered.groupby('Cluster_KMeans_Name')['Monetary'].sum().sort_values(ascending=False)
        
        fig = px.bar(
            x=revenue_by_cluster.index,
            y=revenue_by_cluster.values,
            title="Total Revenue by Segment",
            labels={'x': 'Segment', 'y': 'Revenue (£)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # RFMC Summary Statistics
    st.subheader("RFMC Summary Statistics")
    summary_stats = rfmc_clustered[['Recency', 'Frequency', 'Monetary', 'CancellationRate']].describe()
    st.dataframe(summary_stats)

def show_cluster_analysis(rfmc_clustered):
    st.header("🔍 Cluster Analysis")
    
    # Algorithm selection
    algorithm = st.selectbox(
        "Select Clustering Algorithm:",
        ["K-Means", "Hierarchical", "GMM"] + (["SOM"] if MINISOM_AVAILABLE else [])
    )
    
    cluster_col = f"Cluster_{algorithm.replace('-', '')}"
    cluster_name_col = f"Cluster_{algorithm.replace('-', '')}_Name"
    
    # Cluster profiles
    st.subheader(f"Cluster Profiles - {algorithm}")
    
    features = ["Recency", "Frequency", "Monetary", "CancellationRate"]
    cluster_profiles = rfmc_clustered.groupby(cluster_name_col)[features].mean()
    
    # Display as heatmap
    fig = px.imshow(
        cluster_profiles.T,
        x=cluster_profiles.index,
        y=cluster_profiles.columns,
        color_continuous_scale="RdYlBu_r",
        title=f"Cluster Profiles Heatmap - {algorithm}",
        text_auto=".2f"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed cluster statistics
    st.subheader("Detailed Cluster Statistics")
    
    for cluster_name in cluster_profiles.index:
        with st.expander(f"📋 {cluster_name}"):
            cluster_data = rfmc_clustered[rfmc_clustered[cluster_name_col] == cluster_name]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Customers", len(cluster_data))
            with col2:
                st.metric("Avg Recency", f"{cluster_data['Recency'].mean():.1f} days")
            with col3:
                st.metric("Avg Frequency", f"{cluster_data['Frequency'].mean():.1f}")
            with col4:
                st.metric("Avg Monetary", f"£{cluster_data['Monetary'].mean():.2f}")
            
            # Recommendations based on cluster characteristics
            st.markdown("**💡 Recommendations:**")
            recommendations = get_cluster_recommendations(cluster_name, cluster_data)
            for rec in recommendations:
                st.write(f"• {rec}")
    
    # Scatter plots
    st.subheader("Cluster Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(
            rfmc_clustered,
            x="Recency",
            y="Monetary",
            color=cluster_name_col,
            title="Recency vs Monetary",
            hover_data=["CustomerID", "Frequency"]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            rfmc_clustered,
            x="Frequency",
            y="Monetary",
            color=cluster_name_col,
            title="Frequency vs Monetary",
            hover_data=["CustomerID", "Recency"]
        )
        st.plotly_chart(fig, use_container_width=True)

def show_individual_analysis(rfmc_clustered, data):
    st.header("👤 Individual Customer Analysis")
    
    # Customer selection
    customer_ids = sorted(rfmc_clustered['CustomerID'].unique())
    selected_customer = st.selectbox(
        "Select a Customer ID:",
        customer_ids,
        help="Choose a customer to analyze their RFM profile and purchase behavior"
    )
    
    if selected_customer:
        # Get customer data
        customer_rfmc = rfmc_clustered[rfmc_clustered['CustomerID'] == selected_customer].iloc[0]
        customer_transactions = data[data['Customer ID'] == selected_customer]
        
        # Customer summary
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🎯 Customer Profile")
            
            col1_1, col1_2 = st.columns(2)
            with col1_1:
                st.metric("Customer ID", selected_customer)
                st.metric("Recency", f"{customer_rfmc['Recency']} days")
                st.metric("Frequency", f"{customer_rfmc['Frequency']} orders")
            
            with col1_2:
                st.metric("Monetary Value", f"£{customer_rfmc['Monetary']:.2f}")
                st.metric("Cancellation Rate", f"{customer_rfmc['CancellationRate']:.2%}")
                st.metric("Segment (K-Means)", customer_rfmc['Cluster_KMeans_Name'])
        
        with col2:
            st.subheader("📊 RFM Score Visualization")
            
            # Create radar chart for RFM
            categories = ['Recency\n(Lower is Better)', 'Frequency', 'Monetary', 'Reliability\n(Lower Cancellation)']
            
            # Normalize values for radar chart (0-100 scale)
            recency_score = max(0, 100 - (customer_rfmc['Recency'] / rfmc_clustered['Recency'].max()) * 100)
            frequency_score = (customer_rfmc['Frequency'] / rfmc_clustered['Frequency'].max()) * 100
            monetary_score = (customer_rfmc['Monetary'] / rfmc_clustered['Monetary'].max()) * 100
            reliability_score = max(0, 100 - (customer_rfmc['CancellationRate']) * 100)
            
            values = [recency_score, frequency_score, monetary_score, reliability_score]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=f'Customer {selected_customer}'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True,
                title="Customer RFM Profile"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Transaction history
        st.subheader("🛒 Transaction History")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", len(customer_transactions))
        with col2:
            st.metric("First Purchase", customer_transactions['InvoiceDate'].min().strftime('%Y-%m-%d'))
        with col3:
            st.metric("Last Purchase", customer_transactions['InvoiceDate'].max().strftime('%Y-%m-%d'))
        with col4:
            st.metric("Avg Transaction", f"£{customer_transactions['TotalAmount'].mean():.2f}")
        
        # Purchase timeline
        monthly_purchases = customer_transactions.groupby(
            customer_transactions['InvoiceDate'].dt.to_period('M')
        )['TotalAmount'].sum().reset_index()
        monthly_purchases['InvoiceDate'] = monthly_purchases['InvoiceDate'].astype(str)
        
        fig = px.line(
            monthly_purchases,
            x='InvoiceDate',
            y='TotalAmount',
            title='Monthly Purchase Timeline',
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent transactions table
        st.subheader("Recent Transactions")
        recent_transactions = customer_transactions.sort_values('InvoiceDate', ascending=False).head(10)
        display_cols = ['InvoiceDate', 'Invoice', 'Description', 'Quantity', 'Price', 'TotalAmount']
        st.dataframe(recent_transactions[display_cols])
        
        # Customer insights
        st.subheader("💡 Customer Insights")
        insights = generate_customer_insights(customer_rfmc, customer_transactions)
        for insight in insights:
            st.info(insight)

def show_model_performance(results):
    st.header("📈 Model Performance Comparison")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results).T
    
    # Display metrics table
    st.subheader("Clustering Evaluation Metrics")
    st.dataframe(results_df.round(4))
    
    # Visualize metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = px.bar(
            x=results_df.index,
            y=results_df['silhouette'],
            title="Silhouette Score (Higher is Better)",
            labels={'x': 'Algorithm', 'y': 'Silhouette Score'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            x=results_df.index,
            y=results_df['davies_bouldin'],
            title="Davies-Bouldin Score (Lower is Better)",
            labels={'x': 'Algorithm', 'y': 'Davies-Bouldin Score'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = px.bar(
            x=results_df.index,
            y=results_df['calinski_harabasz'],
            title="Calinski-Harabasz Score (Higher is Better)",
            labels={'x': 'Algorithm', 'y': 'Calinski-Harabasz Score'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Best algorithm recommendation
    st.subheader("🏆 Algorithm Recommendation")
    
    # Normalize scores (higher is better for all)
    normalized_results = results_df.copy()
    normalized_results['silhouette_norm'] = results_df['silhouette']
    normalized_results['davies_bouldin_norm'] = 1 / results_df['davies_bouldin']  # Invert since lower is better
    normalized_results['calinski_harabasz_norm'] = results_df['calinski_harabasz'] / results_df['calinski_harabasz'].max()
    
    # Calculate composite score
    normalized_results['composite_score'] = (
        normalized_results['silhouette_norm'] + 
        normalized_results['davies_bouldin_norm'] + 
        normalized_results['calinski_harabasz_norm']
    ) / 3
    
    best_algorithm = normalized_results['composite_score'].idxmax()
    
    st.success(f"**Recommended Algorithm: {best_algorithm}**")
    st.write(f"Composite Score: {normalized_results.loc[best_algorithm, 'composite_score']:.4f}")

def get_cluster_recommendations(cluster_name, cluster_data):
    """Generate recommendations based on cluster characteristics"""
    recommendations = []
    
    avg_recency = cluster_data['Recency'].mean()
    avg_frequency = cluster_data['Frequency'].mean()
    avg_monetary = cluster_data['Monetary'].mean()
    avg_cancellation = cluster_data['CancellationRate'].mean()
    
    if cluster_name == "Champions":
        recommendations = [
            "Reward loyalty with exclusive offers and early access to new products",
            "Create VIP experiences and premium service tiers",
            "Use them as brand ambassadors and referral sources",
            "Gather feedback for product development"
        ]
    elif cluster_name == "Loyal Regulars":
        recommendations = [
            "Maintain engagement with regular communication",
            "Offer loyalty points and tier upgrades",
            "Cross-sell and upsell complementary products",
            "Provide consistent, reliable service"
        ]
    elif cluster_name == "At Risk":
        recommendations = [
            "Implement win-back campaigns with special discounts",
            "Send personalized re-engagement emails",
            "Offer customer service outreach to address issues",
            "Provide limited-time offers to encourage return"
        ]
    elif cluster_name == "Inactive":
        recommendations = [
            "Launch aggressive win-back campaigns",
            "Offer significant discounts or free shipping",
            "Conduct surveys to understand why they left",
            "Consider these customers for remarketing ads"
        ]
    
    return recommendations

def generate_customer_insights(customer_rfmc, customer_transactions):
    """Generate insights about individual customer"""
    insights = []
    
    # Recency insight
    if customer_rfmc['Recency'] <= 30:
        insights.append("🟢 Recent customer - actively engaged")
    elif customer_rfmc['Recency'] <= 90:
        insights.append("🟡 Moderate recency - may need re-engagement")
    else:
        insights.append("🔴 High recency - at risk of churning")
    
    # Frequency insight
    if customer_rfmc['Frequency'] >= 10:
        insights.append("🟢 High frequency customer - very loyal")
    elif customer_rfmc['Frequency'] >= 5:
        insights.append("🟡 Moderate frequency - regular customer")
    else:
        insights.append("🔴 Low frequency - occasional buyer")
    
    # Monetary insight
    if customer_rfmc['Monetary'] >= customer_transactions.groupby('Customer ID')['TotalAmount'].sum().quantile(0.8):
        insights.append("💰 High-value customer - prioritize retention")
    elif customer_rfmc['Monetary'] >= customer_transactions.groupby('Customer ID')['TotalAmount'].sum().quantile(0.5):
        insights.append("💵 Medium-value customer - potential for growth")
    else:
        insights.append("💴 Lower-value customer - focus on increasing order value")
    
    # Cancellation insight
    if customer_rfmc['CancellationRate'] > 0.1:
        insights.append("⚠️ High cancellation rate - investigate potential issues")
    else:
        insights.append("✅ Low cancellation rate - reliable customer")
    
    return insights

if __name__ == "__main__":
    main()