import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Automated Retail Data Processor & EDA",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.8rem;
        color: #1B4F72;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-left: 4px solid #2E86C1;
        padding-left: 15px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #EBF5FB;
        border-left: 4px solid #2E86C1;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #D5F4E6;
        border-left: 4px solid #27AE60;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FDF2E9;
        border-left: 4px solid #E67E22;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .processing-status {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Create datasets directory if it doesn't exist
def create_datasets_dir():
    """Create datasets directory if it doesn't exist"""
    if not os.path.exists("datasets"):
        os.makedirs("datasets")

class AutomatedDataProcessor:
    """Class to handle automated data processing operations"""
    
    def __init__(self):
        self.df = None
        self.df_cleaned = None
        self.monthly_data = None
        self.weekly_data = None
        self.processing_status = {
            'loaded': False,
            'saved': False,
            'cleaned': False,
            'aggregated': False,
            'all_saved': False
        }
        
    def load_data(self, uploaded_file):
        """Load uploaded CSV file"""
        try:
            self.df = pd.read_csv(uploaded_file)
            self.processing_status['loaded'] = True
            return True
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return False
    
    def save_to_datasets(self):
        """Save the uploaded file to datasets folder as online_retail.csv"""
        try:
            create_datasets_dir()
            self.df.to_csv("datasets/online_retail.csv", index=False)
            self.processing_status['saved'] = True
            return True
        except Exception as e:
            st.error(f"Error saving file: {e}")
            return False
    
    def fill_missing_descriptions(self, show_progress=True):
        """Fill missing product descriptions using mode of same StockCode"""
        df_copy = self.df.copy()
        
        if show_progress:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        iteration = 0
        total_missing_start = df_copy['Description'].isna().sum()
        
        if show_progress:
            status_text.text(f"Processing {total_missing_start} missing descriptions...")
        
        while df_copy['Description'].isna().any():
            iteration += 1
            if iteration > 100:  # Safety break
                if show_progress:
                    st.warning("Maximum iterations reached. Some descriptions may remain unfilled.")
                break
                
            missing_mask = df_copy['Description'].isna()
            missing_stockcodes = df_copy[missing_mask]['StockCode'].unique()
            
            rows_to_remove = []
            filled_count = 0
            
            for i, stockcode in enumerate(missing_stockcodes):
                if show_progress:
                    progress = min((i + 1) / len(missing_stockcodes) * (iteration / 10), 1.0)
                    progress_bar.progress(progress)
                
                stockcode_descriptions = df_copy[
                    (df_copy['StockCode'] == stockcode) & 
                    (df_copy['Description'].notna())
                ]['Description']
                
                if len(stockcode_descriptions) == 0:
                    rows_to_remove.extend(
                        df_copy[
                            (df_copy['StockCode'] == stockcode) & 
                            (df_copy['Description'].isna())
                        ].index.tolist()
                    )
                else:
                    description_counts = stockcode_descriptions.value_counts()
                    mode_description = description_counts.index[0]
                    
                    mask = (df_copy['StockCode'] == stockcode) & (df_copy['Description'].isna())
                    df_copy.loc[mask, 'Description'] = mode_description
                    filled_count += mask.sum()
            
            if rows_to_remove:
                df_copy = df_copy.drop(rows_to_remove)
            
            current_missing = df_copy['Description'].isna().sum()
            if show_progress:
                status_text.text(f"Iteration {iteration}: {current_missing} missing descriptions remaining")
        
        if show_progress:
            progress_bar.progress(1.0)
            status_text.text("Description filling completed!")
        
        self.df_cleaned = df_copy
        self.processing_status['cleaned'] = True
        
        return {
            'original_missing': total_missing_start,
            'final_missing': self.df_cleaned['Description'].isna().sum(),
            'rows_removed': len(self.df) - len(self.df_cleaned)
        }
    
    def create_aggregated_datasets(self, start_date='01/12/2009', end_date='31/12/2010', show_progress=True):
        """Create monthly and weekly aggregated datasets"""
        if self.df_cleaned is None:
            return False
        
        if show_progress:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        try:
            df_clean = self.df_cleaned.copy()
            
            # Convert dates and create features
            if show_progress:
                status_text.text("Converting dates and creating features...")
            df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'], format='%m/%d/%y %H:%M')
            df_clean['NetPrice'] = df_clean['Quantity'] * df_clean['Price']
            df_clean['YearMonth'] = df_clean['InvoiceDate'].dt.to_period('M')
            df_clean['YearWeek'] = df_clean['InvoiceDate'].dt.to_period('W')
            
            if show_progress:
                progress_bar.progress(0.2)
            
            # Product mapping
            if show_progress:
                status_text.text("Creating product mappings...")
            product_mapping = df_clean.groupby('StockCode')['Description'].agg(
                lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]
            ).reset_index()
            
            if show_progress:
                progress_bar.progress(0.4)
            
            # Date ranges
            start_dt = datetime.strptime(start_date, '%d/%m/%Y')
            end_dt = datetime.strptime(end_date, '%d/%m/%Y')
            all_months = pd.period_range(start=start_dt, end=end_dt, freq='M')
            all_weeks = pd.period_range(start=start_dt, end=end_dt, freq='W')
            
            # Monthly aggregation
            if show_progress:
                status_text.text("Creating monthly aggregation...")
            monthly_existing = df_clean.groupby(['StockCode', 'YearMonth']).agg({
                'Quantity': 'sum',
                'NetPrice': 'sum'
            }).reset_index()
            
            monthly_grid = pd.MultiIndex.from_product([
                product_mapping['StockCode'].unique(),
                all_months
            ], names=['StockCode', 'YearMonth']).to_frame(index=False)
            
            monthly_agg = monthly_grid.merge(monthly_existing, on=['StockCode', 'YearMonth'], how='left')
            monthly_agg['Quantity'] = monthly_agg['Quantity'].fillna(0)
            monthly_agg['NetPrice'] = monthly_agg['NetPrice'].fillna(0)
            monthly_agg['Monthly_Per_Unit_Price'] = np.where(
                monthly_agg['Quantity'] > 0,
                monthly_agg['NetPrice'] / monthly_agg['Quantity'],
                0
            )
            
            monthly_agg = monthly_agg.rename(columns={
                'Quantity': 'Monthly_Quantity_Sold',
                'NetPrice': 'Monthly_Total_Price'
            })
            monthly_agg = monthly_agg.merge(product_mapping, on='StockCode', how='left')
            
            if show_progress:
                progress_bar.progress(0.7)
            
            # Weekly aggregation
            if show_progress:
                status_text.text("Creating weekly aggregation...")
            weekly_existing = df_clean.groupby(['StockCode', 'YearWeek']).agg({
                'Quantity': 'sum',
                'NetPrice': 'sum'
            }).reset_index()
            
            weekly_grid = pd.MultiIndex.from_product([
                product_mapping['StockCode'].unique(),
                all_weeks
            ], names=['StockCode', 'YearWeek']).to_frame(index=False)
            
            weekly_agg = weekly_grid.merge(weekly_existing, on=['StockCode', 'YearWeek'], how='left')
            weekly_agg['Quantity'] = weekly_agg['Quantity'].fillna(0)
            weekly_agg['NetPrice'] = weekly_agg['NetPrice'].fillna(0)
            weekly_agg['Weekly_Per_Unit_Price'] = np.where(
                weekly_agg['Quantity'] > 0,
                weekly_agg['NetPrice'] / weekly_agg['Quantity'],
                0
            )
            
            weekly_agg = weekly_agg.rename(columns={
                'Quantity': 'Weekly_Quantity_Sold',
                'NetPrice': 'Weekly_Total_Price'
            })
            weekly_agg = weekly_agg.merge(product_mapping, on='StockCode', how='left')
            
            if show_progress:
                progress_bar.progress(0.9)
            
            # Remove unnecessary stock codes
            if show_progress:
                status_text.text("Removing unnecessary datapoints...")
            unnecessary_values = [
                'DOT', 'D', 'C2', 'BANK CHARGES', 'B', 'AMAZONFEE', 
                'ADJUST2', 'ADJUST', 'S', 'POST', 'M', 'm',
                'gift_0001_10', 'gift_0001_20', 'gift_0001_30', 'gift_0001_40',
                'gift_0001_50', 'gift_0001_60', 'gift_0001_70', 'gift_0001_80'
            ]
            
            monthly_agg = monthly_agg[~monthly_agg['StockCode'].isin(unnecessary_values)]
            weekly_agg = weekly_agg[~weekly_agg['StockCode'].isin(unnecessary_values)]
            
            self.monthly_data = monthly_agg
            self.weekly_data = weekly_agg
            self.processing_status['aggregated'] = True
            
            if show_progress:
                progress_bar.progress(1.0)
                status_text.text("Aggregation completed successfully!")
            
            return True
            
        except Exception as e:
            if show_progress:
                st.error(f"Error in aggregation: {e}")
            return False
    
    def save_all_processed_data(self):
        """Save all processed datasets"""
        try:
            create_datasets_dir()
            
            if self.df_cleaned is not None:
                self.df_cleaned.to_csv("datasets/cleaned_data_with_filled_descriptions.csv", index=False)
                
            if self.monthly_data is not None:
                self.monthly_data.to_csv("datasets/monthly_sales_aggregated.csv", index=False)
                
            if self.weekly_data is not None:
                self.weekly_data.to_csv("datasets/weekly_sales_aggregated.csv", index=False)
                
            self.processing_status['all_saved'] = True
            return True
            
        except Exception as e:
            st.error(f"Error saving processed data: {e}")
            return False
    
    def run_full_pipeline(self, start_date='01/12/2009', end_date='31/12/2010'):
        """Run the complete automated processing pipeline"""
        pipeline_results = {}
        
        # Step 1: Save original data
        st.info("Step 1/4: Saving original data...")
        if self.save_to_datasets():
            pipeline_results['save_original'] = "✅ Original data saved successfully"
        else:
            pipeline_results['save_original'] = "❌ Failed to save original data"
            return pipeline_results
        
        # Step 2: Fill missing descriptions
        st.info("Step 2/4: Processing missing descriptions...")
        cleaning_results = self.fill_missing_descriptions(show_progress=True)
        if self.processing_status['cleaned']:
            pipeline_results['cleaning'] = f"✅ Descriptions processed - {cleaning_results['original_missing']} missing → {cleaning_results['final_missing']} remaining"
            pipeline_results['cleaning_details'] = cleaning_results
        else:
            pipeline_results['cleaning'] = "❌ Failed to process descriptions"
            return pipeline_results
        
        # Step 3: Create aggregations
        st.info("Step 3/4: Creating aggregated datasets...")
        if self.create_aggregated_datasets(start_date, end_date, show_progress=True):
            pipeline_results['aggregation'] = f"✅ Aggregated datasets created - Monthly: {len(self.monthly_data):,} records, Weekly: {len(self.weekly_data):,} records"
        else:
            pipeline_results['aggregation'] = "❌ Failed to create aggregated datasets"
            return pipeline_results
        
        # Step 4: Save all processed data
        st.info("Step 4/4: Saving all processed datasets...")
        if self.save_all_processed_data():
            pipeline_results['save_all'] = "✅ All processed datasets saved successfully"
        else:
            pipeline_results['save_all'] = "❌ Failed to save processed datasets"
            return pipeline_results
        
        pipeline_results['status'] = 'completed'
        return pipeline_results

class EDAAnalyzer:
    """Class to handle EDA operations"""
    
    def __init__(self, df):
        self.df = df
        
    def basic_info(self):
        """Display basic dataset information"""
        st.markdown('<div class="section-header">📋 Dataset Overview</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'<div class="metric-card">Rows<br><strong>{len(self.df):,}</strong></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card">Columns<br><strong>{len(self.df.columns)}</strong></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card">Unique Products<br><strong>{self.df["StockCode"].nunique():,}</strong></div>', unsafe_allow_html=True)
        with col4:
            if 'Customer ID' in self.df.columns:
                st.markdown(f'<div class="metric-card">Unique Customers<br><strong>{self.df["Customer ID"].nunique():,}</strong></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="metric-card">Date Range<br><strong>Analysis Ready</strong></div>', unsafe_allow_html=True)
        
        # Data types and missing values
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Types")
            dtype_df = pd.DataFrame({
                'Column': self.df.columns,
                'Data Type': [str(dtype) for dtype in self.df.dtypes],
                'Non-Null Count': [self.df[col].count() for col in self.df.columns]
            })
            st.dataframe(dtype_df, use_container_width=True)
        
        with col2:
            st.subheader("Missing Values")
            missing_df = pd.DataFrame({
                'Column': self.df.columns,
                'Missing Count': [self.df[col].isnull().sum() for col in self.df.columns],
                'Missing %': [self.df[col].isnull().sum() / len(self.df) * 100 for col in self.df.columns]
            })
            missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
            
            if len(missing_df) > 0:
                st.dataframe(missing_df, use_container_width=True)
                
                # Visualize missing values
                fig = px.bar(missing_df, x='Column', y='Missing %', 
                           title='Missing Values by Column (%)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No missing values found!")
    
    def descriptive_statistics(self):
        """Display descriptive statistics"""
        st.markdown('<div class="section-header">📊 Descriptive Statistics</div>', unsafe_allow_html=True)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            st.subheader("Numerical Columns Summary")
            st.dataframe(self.df[numeric_cols].describe(), use_container_width=True)
            
            # Distribution plots
            st.subheader("Distribution Analysis")
            selected_col = st.selectbox("Select column for distribution analysis:", numeric_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram
                fig = px.histogram(self.df, x=selected_col, nbins=50, 
                                 title=f'Distribution of {selected_col}')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot
                fig = px.box(self.df, y=selected_col, 
                           title=f'Box Plot of {selected_col}')
                st.plotly_chart(fig, use_container_width=True)
    
    def temporal_analysis(self):
        """Perform temporal analysis"""
        st.markdown('<div class="section-header">📅 Temporal Analysis</div>', unsafe_allow_html=True)
        
        # Check for date columns
        date_cols = []
        for col in self.df.columns:
            if 'date' in col.lower() or 'Date' in col or any(x in col for x in ['YearMonth', 'YearWeek']):
                date_cols.append(col)
        
        if not date_cols:
            st.warning("No date columns found for temporal analysis")
            return
        
        # For aggregated data with YearMonth or YearWeek
        if 'YearMonth' in self.df.columns or 'YearWeek' in self.df.columns:
            if 'YearMonth' in self.df.columns:
                # Monthly analysis
                monthly_totals = self.df.groupby('YearMonth')['Monthly_Total_Price'].sum().reset_index()
                monthly_totals['YearMonth_str'] = monthly_totals['YearMonth'].astype(str)
                
                fig = px.line(monthly_totals, x='YearMonth_str', y='Monthly_Total_Price',
                            title='Monthly Sales Trend')
                fig.update_xaxis(title='Month')
                st.plotly_chart(fig, use_container_width=True)
            
            if 'YearWeek' in self.df.columns:
                # Weekly analysis
                weekly_totals = self.df.groupby('YearWeek')['Weekly_Total_Price'].sum().reset_index()
                weekly_totals['YearWeek_str'] = weekly_totals['YearWeek'].astype(str)
                
                fig = px.line(weekly_totals, x='YearWeek_str', y='Weekly_Total_Price',
                            title='Weekly Sales Trend')
                fig.update_xaxis(title='Week')
                st.plotly_chart(fig, use_container_width=True)
        
        # For original data with InvoiceDate
        elif 'InvoiceDate' in self.df.columns:
            df_temp = self.df.copy()
            try:
                df_temp['InvoiceDate'] = pd.to_datetime(df_temp['InvoiceDate'])
                df_temp['NetPrice'] = df_temp['Quantity'] * df_temp['Price']
                
                # Extract temporal features
                df_temp['Month'] = df_temp['InvoiceDate'].dt.month
                df_temp['DayOfWeek'] = df_temp['InvoiceDate'].dt.day_name()
                df_temp['Hour'] = df_temp['InvoiceDate'].dt.hour
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Daily sales
                    daily_sales = df_temp.groupby(df_temp['InvoiceDate'].dt.date)['NetPrice'].sum().reset_index()
                    fig = px.line(daily_sales, x='InvoiceDate', y='NetPrice',
                                title='Daily Sales Trend')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Monthly sales
                    monthly_sales = df_temp.groupby('Month')['NetPrice'].sum().reset_index()
                    fig = px.bar(monthly_sales, x='Month', y='NetPrice',
                               title='Sales by Month')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Day of week analysis
                    dow_sales = df_temp.groupby('DayOfWeek')['NetPrice'].sum().reset_index()
                    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    dow_sales['DayOfWeek'] = pd.Categorical(dow_sales['DayOfWeek'], categories=dow_order, ordered=True)
                    dow_sales = dow_sales.sort_values('DayOfWeek')
                    
                    fig = px.bar(dow_sales, x='DayOfWeek', y='NetPrice',
                               title='Sales by Day of Week')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Hourly analysis
                    hourly_sales = df_temp.groupby('Hour')['NetPrice'].sum().reset_index()
                    fig = px.line(hourly_sales, x='Hour', y='NetPrice',
                                title='Sales by Hour of Day')
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error in temporal analysis: {e}")
    
    def product_analysis(self):
        """Analyze product performance"""
        st.markdown('<div class="section-header">🛍️ Product Analysis</div>', unsafe_allow_html=True)
        
        # Check if this is aggregated data
        if 'Monthly_Total_Price' in self.df.columns:
            # Monthly aggregated data analysis
            product_metrics = self.df.groupby(['StockCode', 'Description']).agg({
                'Monthly_Total_Price': 'sum',
                'Monthly_Quantity_Sold': 'sum'
            }).reset_index()
            
            revenue_col = 'Monthly_Total_Price'
            quantity_col = 'Monthly_Quantity_Sold'
            
        elif 'Weekly_Total_Price' in self.df.columns:
            # Weekly aggregated data analysis
            product_metrics = self.df.groupby(['StockCode', 'Description']).agg({
                'Weekly_Total_Price': 'sum',
                'Weekly_Quantity_Sold': 'sum'
            }).reset_index()
            
            revenue_col = 'Weekly_Total_Price'
            quantity_col = 'Weekly_Quantity_Sold'
            
        else:
            # Original data analysis
            df_temp = self.df.copy()
            df_temp['NetPrice'] = df_temp['Quantity'] * df_temp['Price']
            
            product_metrics = df_temp.groupby(['StockCode', 'Description']).agg({
                'Invoice': 'nunique',
                'NetPrice': 'sum',
                'Quantity': 'sum'
            }).reset_index()
            
            product_metrics.columns = ['StockCode', 'Description', 'Num_Orders', 'Total_Revenue', 'Total_Quantity']
            revenue_col = 'Total_Revenue'
            quantity_col = 'Total_Quantity'
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top 10 Products by Revenue")
            top_revenue = product_metrics.nlargest(10, revenue_col)[['StockCode', 'Description', revenue_col]]
            
            fig = px.bar(top_revenue, x='StockCode', y=revenue_col,
                       title='Top Products by Revenue',
                       hover_data=['Description'])
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(top_revenue, use_container_width=True)
        
        with col2:
            st.subheader("Top 10 Products by Quantity")
            top_quantity = product_metrics.nlargest(10, quantity_col)[['StockCode', 'Description', quantity_col]]
            
            fig = px.bar(top_quantity, x='StockCode', y=quantity_col,
                       title='Top Products by Quantity Sold',
                       hover_data=['Description'])
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(top_quantity, use_container_width=True)
    
    def correlation_analysis(self):
        """Perform correlation analysis"""
        st.markdown('<div class="section-header">🔗 Correlation Analysis</div>', unsafe_allow_html=True)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            correlation_matrix = self.df[numeric_cols].corr()
            
            fig = px.imshow(correlation_matrix, 
                          text_auto=True,
                          aspect="auto",
                          title="Correlation Heatmap",
                          color_continuous_scale='RdBu')
            st.plotly_chart(fig, use_container_width=True)
            
            # Show correlation table
            st.subheader("Correlation Matrix")
            st.dataframe(correlation_matrix, use_container_width=True)
        else:
            st.warning("Not enough numerical columns for correlation analysis")

def main():
    st.markdown('<div class="main-header">🤖 Automated Retail Data Processor & EDA</div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'processor' not in st.session_state:
        st.session_state.processor = AutomatedDataProcessor()
    if 'auto_processing_complete' not in st.session_state:
        st.session_state.auto_processing_complete = False
    if 'pipeline_results' not in st.session_state:
        st.session_state.pipeline_results = None
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    # Show processing status in sidebar
    if st.session_state.processor.processing_status['loaded']:
        st.sidebar.markdown("### Processing Status")
        status = st.session_state.processor.processing_status
        st.sidebar.write(f"✅ Data Loaded" if status['loaded'] else "⏸️ Data Loaded")
        st.sidebar.write(f"✅ Data Saved" if status['saved'] else "⏸️ Data Saved")
        st.sidebar.write(f"✅ Data Cleaned" if status['cleaned'] else "⏸️ Data Cleaned")
        st.sidebar.write(f"✅ Data Aggregated" if status['aggregated'] else "⏸️ Data Aggregated")
        st.sidebar.write(f"✅ All Saved" if status['all_saved'] else "⏸️ All Saved")
    
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🚀 Auto Upload & Process", "📊 Exploratory Data Analysis", "💾 Download Results"]
    )
    
    if page == "🚀 Auto Upload & Process":
        show_automated_upload_page()
    elif page == "📊 Exploratory Data Analysis":
        show_eda_page()
    elif page == "💾 Download Results":
        show_download_page()

def show_automated_upload_page():
    """Show automated upload and processing interface"""
    st.header("🚀 Automated Data Upload & Processing")
    
    st.markdown("""
    <div class="info-box">
    <h4>🤖 Fully Automated Processing Pipeline</h4>
    <p>This streamlined version automatically processes your data through all steps:</p>
    <ul>
        <li><strong>Upload & Save:</strong> CSV file saved to datasets folder</li>
        <li><strong>Auto Clean:</strong> Missing descriptions filled using statistical methods</li>
        <li><strong>Auto Aggregate:</strong> Monthly and weekly sales summaries created</li>
        <li><strong>Auto Export:</strong> All processed datasets saved automatically</li>
    </ul>
    <p><em>Simply upload your file and let the system do the rest!</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose your retail CSV file",
        type="csv",
        help="Upload your retail transactions CSV file - processing will start automatically"
    )
    
    # Processing parameters
    with st.expander("⚙️ Processing Options (Optional)"):
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date for Aggregation", value=datetime(2009, 12, 1))
        with col2:
            end_date = st.date_input("End Date for Aggregation", value=datetime(2010, 12, 31))
    
    if uploaded_file is not None:
        # Load the data first
        if not st.session_state.processor.processing_status['loaded']:
            with st.spinner("Loading your data..."):
                if st.session_state.processor.load_data(uploaded_file):
                    st.success(f"✅ Data loaded successfully! Shape: {st.session_state.processor.df.shape}")
                    
                    # Show quick preview
                    with st.expander("📋 Data Preview"):
                        st.dataframe(st.session_state.processor.df.head(), use_container_width=True)
                        
                        # Quick stats
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Records", f"{len(st.session_state.processor.df):,}")
                        with col2:
                            st.metric("Unique Products", f"{st.session_state.processor.df['StockCode'].nunique():,}")
                        with col3:
                            if 'Customer ID' in st.session_state.processor.df.columns:
                                st.metric("Unique Customers", f"{st.session_state.processor.df['Customer ID'].nunique():,}")
                            else:
                                st.metric("Columns", len(st.session_state.processor.df.columns))
                        with col4:
                            missing_desc = st.session_state.processor.df['Description'].isnull().sum()
                            st.metric("Missing Descriptions", f"{missing_desc:,}")
        
        # Auto-process button or status
        if st.session_state.processor.processing_status['loaded'] and not st.session_state.auto_processing_complete:
            st.markdown("### 🚀 Ready for Automated Processing")
            
            if st.button("🤖 Run Complete Auto-Processing Pipeline", type="primary", use_container_width=True):
                start_str = start_date.strftime('%d/%m/%Y')
                end_str = end_date.strftime('%d/%m/%Y')
                
                # Create processing status container
                status_container = st.container()
                
                with status_container:
                    st.markdown("### 🔄 Processing Status")
                    
                    # Run the automated pipeline
                    with st.spinner("Running automated processing pipeline..."):
                        pipeline_results = st.session_state.processor.run_full_pipeline(start_str, end_str)
                        st.session_state.pipeline_results = pipeline_results
                    
                    # Show results
                    if pipeline_results.get('status') == 'completed':
                        st.session_state.auto_processing_complete = True
                        
                        st.markdown("""
                        <div class="success-box">
                        <h3>🎉 Automated Processing Completed Successfully!</h3>
                        <p>All processing steps have been completed automatically. Your data is now ready for analysis!</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show detailed results
                        st.markdown("### 📊 Processing Summary")
                        for key, message in pipeline_results.items():
                            if key != 'status' and key != 'cleaning_details':
                                st.write(message)
                        
                        # Show cleaning details if available
                        if 'cleaning_details' in pipeline_results:
                            cleaning = pipeline_results['cleaning_details']
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Original Missing", f"{cleaning['original_missing']:,}")
                            with col2:
                                st.metric("Final Missing", f"{cleaning['final_missing']:,}")
                            with col3:
                                st.metric("Rows Removed", f"{cleaning['rows_removed']:,}")
                        
                        # Dataset summary
                        st.markdown("### 📁 Generated Datasets")
                        datasets_info = [
                            ("Original Dataset", "datasets/online_retail.csv", len(st.session_state.processor.df)),
                            ("Cleaned Dataset", "datasets/cleaned_data_with_filled_descriptions.csv", len(st.session_state.processor.df_cleaned)),
                            ("Monthly Aggregated", "datasets/monthly_sales_aggregated.csv", len(st.session_state.processor.monthly_data)),
                            ("Weekly Aggregated", "datasets/weekly_sales_aggregated.csv", len(st.session_state.processor.weekly_data))
                        ]
                        
                        for name, path, rows in datasets_info:
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"📄 **{name}** - {rows:,} records")
                            with col2:
                                if os.path.exists(path):
                                    st.write("✅ Saved")
                        
                        st.balloons()
                        
                        # Next steps
                        st.markdown("""
                        <div class="info-box">
                        <h4>🎯 What's Next?</h4>
                        <ul>
                            <li><strong>Explore Data:</strong> Navigate to "Exploratory Data Analysis" to analyze your processed data</li>
                            <li><strong>Download Files:</strong> Go to "Download Results" to get all processed datasets</li>
                            <li><strong>Advanced Analysis:</strong> Use the cleaned data for machine learning or statistical modeling</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    else:
                        st.error("❌ Processing failed. Please check your data format and try again.")
                        for key, message in pipeline_results.items():
                            if "❌" in str(message):
                                st.write(message)
        
        elif st.session_state.auto_processing_complete:
            # Show completion status
            st.markdown("""
            <div class="processing-status">
            <h3>✅ Processing Complete</h3>
            <p>Your data has been successfully processed through the entire pipeline!</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Processing summary
            if st.session_state.pipeline_results:
                with st.expander("📋 View Processing Summary"):
                    for key, message in st.session_state.pipeline_results.items():
                        if key != 'status' and key != 'cleaning_details':
                            st.write(message)
            
            # Quick actions
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 Go to EDA", type="secondary", use_container_width=True):
                    st.experimental_rerun()
            with col2:
                if st.button("💾 Download Results", type="secondary", use_container_width=True):
                    st.experimental_rerun()

def show_eda_page():
    """Show EDA interface"""
    st.header("📊 Exploratory Data Analysis")
    
    if not st.session_state.processor.processing_status['loaded']:
        st.warning("⚠️ No data loaded. Please upload and process your data first!")
        return
    
    # Check available datasets
    data_options = {}
    
    if st.session_state.processor.df is not None:
        data_options["Original Dataset"] = st.session_state.processor.df
    
    if st.session_state.processor.df_cleaned is not None:
        data_options["Cleaned Dataset"] = st.session_state.processor.df_cleaned
    
    if st.session_state.processor.monthly_data is not None:
        data_options["Monthly Aggregated"] = st.session_state.processor.monthly_data
    
    if st.session_state.processor.weekly_data is not None:
        data_options["Weekly Aggregated"] = st.session_state.processor.weekly_data
    
    if not data_options:
        st.warning("No datasets available for analysis. Please complete the processing pipeline first!")
        return
    
    # Dataset selection
    st.markdown("### 📂 Select Dataset for Analysis")
    selected_dataset = st.selectbox(
        "Choose dataset:",
        list(data_options.keys()),
        help="Select which processed dataset you want to analyze"
    )
    
    df_selected = data_options[selected_dataset]
    
    # Show dataset info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Records", f"{len(df_selected):,}")
    with col2:
        st.metric("Columns", len(df_selected.columns))
    with col3:
        if 'StockCode' in df_selected.columns:
            st.metric("Products", f"{df_selected['StockCode'].nunique():,}")
    with col4:
        memory_usage = df_selected.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("Memory", f"{memory_usage:.1f} MB")
    
    # Analysis selection
    st.markdown("### 🔍 Analysis Options")
    analysis_options = st.multiselect(
        "Select analyses to perform:",
        [
            "Dataset Overview",
            "Descriptive Statistics", 
            "Temporal Analysis",
            "Product Analysis",
            "Correlation Analysis"
        ],
        default=["Dataset Overview", "Descriptive Statistics"]
    )
    
    # Perform selected analyses
    if analysis_options:
        analyzer = EDAAnalyzer(df_selected)
        
        for analysis in analysis_options:
            try:
                if analysis == "Dataset Overview":
                    analyzer.basic_info()
                elif analysis == "Descriptive Statistics":
                    analyzer.descriptive_statistics()
                elif analysis == "Temporal Analysis":
                    analyzer.temporal_analysis()
                elif analysis == "Product Analysis":
                    analyzer.product_analysis()
                elif analysis == "Correlation Analysis":
                    analyzer.correlation_analysis()
                    
                # Add separator between analyses
                if analysis != analysis_options[-1]:
                    st.markdown("---")
                    
            except Exception as e:
                st.error(f"Error in {analysis}: {str(e)}")
    
    # Advanced filtering section
    with st.expander("🔧 Advanced Data Filtering & Custom Analysis"):
        st.markdown("### Filter Data for Focused Analysis")
        
        # Numeric filters
        numeric_cols = df_selected.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            st.subheader("Numeric Filters")
            selected_numeric = st.selectbox("Filter by numeric column:", ["None"] + numeric_cols)
            
            if selected_numeric != "None":
                col_min, col_max = st.columns(2)
                with col_min:
                    min_val = st.number_input(f"Minimum {selected_numeric}", 
                                            value=float(df_selected[selected_numeric].min()))
                with col_max:
                    max_val = st.number_input(f"Maximum {selected_numeric}", 
                                            value=float(df_selected[selected_numeric].max()))
                
                if st.button("Apply Numeric Filter"):
                    filtered_df = df_selected[
                        (df_selected[selected_numeric] >= min_val) & 
                        (df_selected[selected_numeric] <= max_val)
                    ]
                    st.success(f"✅ Filtered to {len(filtered_df):,} rows (from {len(df_selected):,})")
                    
                    # Show sample of filtered data
                    st.subheader("Filtered Data Sample")
                    st.dataframe(filtered_df.head(10), use_container_width=True)
        
        # Categorical filters
        categorical_cols = df_selected.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            st.subheader("Categorical Filters")
            selected_categorical = st.selectbox("Filter by categorical column:", ["None"] + categorical_cols)
            
            if selected_categorical != "None":
                unique_values = df_selected[selected_categorical].dropna().unique()
                if len(unique_values) <= 50:  # Only show if manageable number of values
                    selected_values = st.multiselect(
                        f"Select {selected_categorical} values:",
                        unique_values,
                        default=list(unique_values[:10]) if len(unique_values) > 10 else list(unique_values)
                    )
                    
                    if st.button("Apply Categorical Filter") and selected_values:
                        filtered_df = df_selected[df_selected[selected_categorical].isin(selected_values)]
                        st.success(f"✅ Filtered to {len(filtered_df):,} rows (from {len(df_selected):,})")
                        
                        # Show sample of filtered data
                        st.subheader("Filtered Data Sample")
                        st.dataframe(filtered_df.head(10), use_container_width=True)
                else:
                    st.info(f"Too many unique values ({len(unique_values)}) to display filter options")

def show_download_page():
    """Show download interface for processed data"""
    st.header("💾 Download Processed Data")
    
    if not st.session_state.auto_processing_complete:
        st.warning("⚠️ Processing not completed yet. Please complete the automated processing first!")
        return
    
    st.markdown("""
    <div class="success-box">
    <h4>✅ All Processing Complete!</h4>
    <p>Your retail data has been successfully processed through the entire automated pipeline. 
    Download the processed datasets below for further analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check available files
    files_info = [
        ("📄 Original Dataset", "datasets/online_retail.csv", "The raw uploaded data as saved to datasets folder"),
        ("🧹 Cleaned Dataset", "datasets/cleaned_data_with_filled_descriptions.csv", "Data with missing descriptions filled using statistical methods"),
        ("📅 Monthly Aggregated", "datasets/monthly_sales_aggregated.csv", "Monthly sales summary with product-level aggregations"),
        ("📊 Weekly Aggregated", "datasets/weekly_sales_aggregated.csv", "Weekly sales summary with product-level aggregations")
    ]
    
    available_files = []
    for name, path, description in files_info:
        if os.path.exists(path):
            file_size = os.path.getsize(path) / 1024 / 1024  # Size in MB
            available_files.append((name, path, description, file_size))
    
    if not available_files:
        st.error("❌ No processed files found. Please complete the processing pipeline first.")
        return
    
    st.success(f"🎉 Found {len(available_files)} processed datasets ready for download!")
    
    # Download section
    st.markdown("### 📥 Download Files")
    
    for name, file_path, description, file_size in available_files:
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**{name}** ({file_size:.1f} MB)")
                st.caption(description)
            
            with col2:
                try:
                    with open(file_path, 'rb') as f:
                        file_data = f.read()
                    
                    st.download_button(
                        label="⬇️ Download",
                        data=file_data,
                        file_name=os.path.basename(file_path),
                        mime="text/csv",
                        key=f"download_{os.path.basename(file_path)}",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Error preparing download: {e}")
    
    # File summaries
    st.markdown("### 📋 Dataset Summaries")
    
    for name, file_path, description, file_size in available_files:
        with st.expander(f"📊 {name} Details"):
            try:
                df_temp = pd.read_csv(file_path, nrows=1000)  # Load sample for summary
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Rows", f"{len(pd.read_csv(file_path)):,}")
                with col2:
                    st.metric("Columns", len(df_temp.columns))
                with col3:
                    st.metric("Size", f"{file_size:.1f} MB")
                with col4:
                    memory_usage = df_temp.memory_usage(deep=True).sum() / 1024 / 1024
                    st.metric("Memory", f"{memory_usage:.1f} MB")
                
                # Column information
                st.subheader("📋 Columns")
                col_info = []
                for col in df_temp.columns:
                    dtype = str(df_temp[col].dtype)
                    non_null = df_temp[col].count()
                    col_info.append([col, dtype, non_null])
                
                col_df = pd.DataFrame(col_info, columns=['Column', 'Data Type', 'Non-Null Count'])
                st.dataframe(col_df, use_container_width=True)
                
                # Data preview
                st.subheader("🔍 Data Preview")
                st.dataframe(df_temp.head(), use_container_width=True)
                
            except Exception as e:
                st.error(f"Error analyzing {name}: {e}")
    
    # Processing summary
    if st.session_state.pipeline_results:
        st.markdown("### 🔄 Processing Summary")
        with st.expander("View Complete Processing Log"):
            for key, message in st.session_state.pipeline_results.items():
                if key != 'status' and key != 'cleaning_details':
                    st.write(message)
    
    # Usage recommendations
    st.markdown("""
    <div class="info-box">
    <h4>💡 Usage Recommendations</h4>
    <ul>
        <li><strong>For Business Intelligence:</strong> Use Monthly/Weekly aggregated data for dashboards and reporting</li>
        <li><strong>For Machine Learning:</strong> Use the cleaned dataset for predictive modeling and recommendations</li>
        <li><strong>For Statistical Analysis:</strong> All datasets are ready for advanced analytics in R, Python, or other tools</li>
        <li><strong>For Visualization:</strong> Import into Tableau, Power BI, or similar tools for interactive dashboards</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()