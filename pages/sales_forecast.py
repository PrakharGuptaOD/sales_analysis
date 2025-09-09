# Import necessary libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Enhanced Sales Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .forecast-container {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .evaluation-metrics {
        background-color: #fff8dc;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffa500;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
# Define function to load and cache the aggregated sales data.
def load_data():
    """Load and cache the aggregated sales data."""
    try:
        # Load monthly data
        if os.path.exists('datasets/monthly_sales_aggregated.csv'):
            monthly_df = pd.read_csv('datasets/monthly_sales_aggregated.csv')
            # Convert YearMonth to datetime for better plotting
            monthly_df['YearMonth'] = pd.to_datetime(monthly_df['YearMonth'])
        else:
            monthly_df = pd.DataFrame()
            st.error("monthly_sales_aggregated.csv not found!")
        
        # Load weekly data
        if os.path.exists('datasets/weekly_sales_aggregated.csv'):
            weekly_df = pd.read_csv('datasets/weekly_sales_aggregated.csv')
            # Convert YearWeek period to datetime - use start of week
            weekly_df['YearWeek_Period'] = weekly_df['YearWeek'].astype(str).apply(pd.Period, freq='W')
            weekly_df['YearWeek'] = weekly_df['YearWeek_Period'].dt.start_time
        else:
            weekly_df = pd.DataFrame()
            st.error("weekly_sales_aggregated.csv not found!")
        
        return monthly_df, weekly_df
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame()

# Define function for simple moving average forecasting
def simple_moving_average_forecast(ts_data, window_size=3, periods=3):
    """
    Simple Moving Average forecasting
    
    Parameters:
    - ts_data: Time series data
    - window_size: Number of periods to use for moving average
    - periods: Number of periods to forecast
    """
    ts_clean = ts_data.dropna()
    
    if len(ts_clean) == 0:
        return [0] * periods, None, None
    
    # Adjust window size if data is too short
    window_size = min(window_size, len(ts_clean))
    
    if window_size == 0:
        return [ts_clean.iloc[-1]] * periods, None, None
    
    # Calculate moving average for the last window_size periods
    recent_values = ts_clean.tail(window_size).values
    sma_forecast = np.mean(recent_values)
    
    # Generate forecasts (constant for SMA)
    forecasts = [sma_forecast] * periods
    
    # Calculate confidence intervals based on historical variance
    if len(ts_clean) > window_size:
        # Calculate rolling standard deviation
        rolling_std = ts_clean.rolling(window=window_size).std().dropna()
        if len(rolling_std) > 0:
            std_dev = rolling_std.mean()
            # 95% confidence interval (approximately 1.96 standard deviations)
            lower_ci = [sma_forecast - 1.96 * std_dev] * periods
            upper_ci = [sma_forecast + 1.96 * std_dev] * periods
            confidence_intervals = {'lower': lower_ci, 'upper': upper_ci}
        else:
            confidence_intervals = None
    else:
        confidence_intervals = None
    
    # Return model info as a dictionary
    model_info = {
        'window_size': window_size,
        'forecast_value': sma_forecast,
        'last_values': recent_values.tolist()
    }
    
    return forecasts, model_info, confidence_intervals


# Define function to calculate various evaluation metrics
def calculate_evaluation_metrics(actual, predicted):
    """Calculate various evaluation metrics"""
    try:
        # Ensure arrays are the same length
        min_length = min(len(actual), len(predicted))
        actual = np.array(actual[:min_length])
        predicted = np.array(predicted[:min_length])
        
        # Remove any NaN or infinite values
        mask = np.isfinite(actual) & np.isfinite(predicted)
        actual = actual[mask]
        predicted = predicted[mask]
        
        if len(actual) == 0:
            return None
        
        # Mean Absolute Error
        mae = np.mean(np.abs(actual - predicted))
        
        # Mean Squared Error
        mse = np.mean((actual - predicted) ** 2)
        
        # Root Mean Squared Error
        rmse = np.sqrt(mse)
        
        # Mean Absolute Percentage Error (with handling for zero values)
        if np.any(actual == 0):
            # Use symmetric MAPE when there are zero values
            mape = np.mean(2 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted) + 1e-8)) * 100
        else:
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        # Mean Percentage Error
        mpe = np.mean((actual - predicted) / (actual + 1e-8)) * 100
        
        # R-squared
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'MPE': mpe,
            'R2': r2,
            'Mean_Actual': np.mean(actual),
            'Mean_Predicted': np.mean(predicted)
        }
    except Exception as e:
        st.error(f"Error calculating metrics: {e}")
        return None

# Define function to split time series data into train and test sets
def train_test_split_ts(ts_data, test_ratio=0.2):
    """Split time series data into train and test sets"""
    ts_clean = ts_data.dropna()
    split_point = int(len(ts_clean) * (1 - test_ratio))
    
    if split_point < 3:  # Ensure minimum training data
        split_point = max(1, len(ts_clean) - 2)
    
    train_data = ts_clean.iloc[:split_point]
    test_data = ts_clean.iloc[split_point:]
    
    return train_data, test_data

# Define function to create product options for dropdown by combining StockCode and Description.
def create_product_options(monthly_df, weekly_df):
    """Create product options for dropdown by combining StockCode and Description."""
    
    # Get unique products from both datasets
    products = set()
    
    if not monthly_df.empty:
        monthly_products = monthly_df[['StockCode', 'Description']].drop_duplicates()
        for _, row in monthly_products.iterrows():
            products.add((row['StockCode'], row['Description']))
    
    if not weekly_df.empty:
        weekly_products = weekly_df[['StockCode', 'Description']].drop_duplicates()
        for _, row in weekly_products.iterrows():
            products.add((row['StockCode'], row['Description']))
    
    # Create formatted options
    product_options = {}
    for stockcode, description in sorted(products):
        display_text = f"{stockcode} - {description}"
        product_options[display_text] = stockcode
    
    return product_options

# Define function to filter dataframe to show only recent N weeks.
def filter_recent_weeks(df, weeks=10):
    """Filter dataframe to show only recent N weeks."""
    if df.empty:
        return df
    
    # Get the latest date and calculate cutoff
    latest_date = df['YearWeek'].max()
    cutoff_date = latest_date - timedelta(weeks=weeks)
    
    return df[df['YearWeek'] >= cutoff_date]

# Define function to create comprehensive forecast visualization
def create_forecast_visualization(data, forecasts, confidence_intervals, analysis_type, metric_type, product_name, window_size=3):
    """Create comprehensive forecast visualization"""
    
    fig = go.Figure()
    
    # Determine columns based on analysis type and metric type
    if analysis_type == 'monthly':
        time_col = 'YearMonth'
        if metric_type == 'quantity':
            value_col = 'Monthly_Quantity_Sold'
            y_title = "Quantity Sold (Units)"
            chart_title = f"Monthly Quantity Forecast (SMA-{window_size}) - {product_name}"
        elif metric_type == 'price':
            value_col = 'Monthly_Per_Unit_Price'
            y_title = "Per Unit Price (£)"
            chart_title = f"Monthly Price Forecast (SMA-{window_size}) - {product_name}"
        else:  # revenue
            value_col = 'Monthly_Total_Price'
            y_title = "Total Revenue (£)"
            chart_title = f"Monthly Revenue Forecast (SMA-{window_size}) - {product_name}"
    else:  # weekly
        time_col = 'YearWeek'
        if metric_type == 'quantity':
            value_col = 'Weekly_Quantity_Sold'
            y_title = "Quantity Sold (Units)"
            chart_title = f"Weekly Quantity Forecast (SMA-{window_size}) - {product_name}"
        elif metric_type == 'price':
            value_col = 'Weekly_Per_Unit_Price'
            y_title = "Per Unit Price (£)"
            chart_title = f"Weekly Price Forecast (SMA-{window_size}) - {product_name}"
        else:  # revenue
            value_col = 'Weekly_Total_Price'
            y_title = "Total Revenue (£)"
            chart_title = f"Weekly Revenue Forecast (SMA-{window_size}) - {product_name}"
    
    if data.empty:
        fig.add_annotation(
            text="No data available for selected product",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle'
        )
        fig.update_layout(title=chart_title, showlegend=False)
        return fig
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=data[time_col],
        y=data[value_col],
        mode='lines+markers',
        name='Historical Data',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6)
    ))
    
    # Add moving average line on historical data
    if len(data) >= window_size:
        ma_values = data[value_col].rolling(window=window_size).mean()
        fig.add_trace(go.Scatter(
            x=data[time_col],
            y=ma_values,
            mode='lines',
            name=f'SMA-{window_size}',
            line=dict(color='#2ca02c', width=2, dash='dot'),
            opacity=0.7
        ))
    
    # Forecast data
    if forecasts and len(forecasts) > 0:
        # Generate future dates
        last_date = data[time_col].max()
        if analysis_type == 'monthly':
            future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(len(forecasts))]
        else:
            future_dates = [last_date + timedelta(weeks=i+1) for i in range(len(forecasts))]
        
        # Forecast line
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=forecasts,
            mode='lines+markers',
            name='SMA Forecast',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            marker=dict(size=8, symbol='diamond')
        ))
        
        # Confidence intervals
        if confidence_intervals and 'lower' in confidence_intervals and 'upper' in confidence_intervals:
            fig.add_trace(go.Scatter(
                x=future_dates + future_dates[::-1],
                y=confidence_intervals['upper'] + confidence_intervals['lower'][::-1],
                fill='toself',
                fillcolor='rgba(255, 127, 14, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence Interval',
                hoverinfo="skip"
            ))
        
        # Connection line between historical and forecast
        fig.add_trace(go.Scatter(
            x=[last_date, future_dates[0]],
            y=[data[value_col].iloc[-1], forecasts[0]],
            mode='lines',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            showlegend=False,
            hoverinfo="skip"
        ))
    
    fig.update_layout(
        title=chart_title,
        xaxis_title="Time Period",
        yaxis_title=y_title,
        hovermode='x unified',
        template='plotly_white',
        showlegend=True
    )
    
    return fig

# Define function to create quantity sold chart.
def create_quantity_chart(data, analysis_type, product_name):
    """Create quantity sold chart."""
    
    if data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for selected product",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle'
        )
        fig.update_layout(
            title=f"{analysis_type.title()} Quantity Sold - {product_name}",
            showlegend=False
        )
        return fig
    
    if analysis_type == 'monthly':
        x_col = 'YearMonth'
        y_col = 'Monthly_Quantity_Sold'
        title = f"Monthly Quantity Sold - {product_name}"
    else:
        x_col = 'YearWeek'
        y_col = 'Weekly_Quantity_Sold'
        title = f"Weekly Quantity Sold (Recent 10 Weeks) - {product_name}"
    
    fig = px.line(
        data, 
        x=x_col, 
        y=y_col,
        title=title,
        markers=True,
        line_shape='spline'
    )
    
    fig.update_layout(
        xaxis_title="Time Period",
        yaxis_title="Quantity Sold (Units)",
        hovermode='x unified',
        showlegend=False
    )
    
    fig.update_traces(
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    )
    
    return fig

# Define function to create per unit price chart.
def create_per_unit_price_chart(data, analysis_type, product_name):
    """Create per unit price chart."""
    
    if data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for selected product",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle'
        )
        fig.update_layout(
            title=f"{analysis_type.title()} Per Unit Price - {product_name}",
            showlegend=False
        )
        return fig
    
    if analysis_type == 'monthly':
        x_col = 'YearMonth'
        y_col = 'Monthly_Per_Unit_Price'
        title = f"Monthly Per Unit Price - {product_name}"
    else:
        x_col = 'YearWeek'
        y_col = 'Weekly_Per_Unit_Price'
        title = f"Weekly Per Unit Price (Recent 10 Weeks) - {product_name}"
    
    fig = px.line(
        data, 
        x=x_col, 
        y=y_col,
        title=title,
        markers=True,
        line_shape='spline'
    )
    
    fig.update_layout(
        xaxis_title="Time Period",
        yaxis_title="Per Unit Price (£)",
        hovermode='x unified',
        showlegend=False,
        yaxis=dict(tickformat='£,.2f')
    )
    
    fig.update_traces(
        line=dict(color='#2ca02c', width=3),
        marker=dict(size=8)
    )
    
    return fig

# Define function to create total price chart.
def create_price_chart(data, analysis_type, product_name):
    """Create total price chart."""
    
    if data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for selected product",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle'
        )
        fig.update_layout(
            title=f"{analysis_type.title()} Total Revenue - {product_name}",
            showlegend=False
        )
        return fig
    
    if analysis_type == 'monthly':
        x_col = 'YearMonth'
        y_col = 'Monthly_Total_Price'
        title = f"Monthly Total Revenue - {product_name}"
    else:
        x_col = 'YearWeek'
        y_col = 'Weekly_Total_Price'
        title = f"Weekly Total Revenue (Recent 10 Weeks) - {product_name}"
    
    fig = px.line(
        data, 
        x=x_col, 
        y=y_col,
        title=title,
        markers=True,
        line_shape='spline'
    )
    
    fig.update_layout(
        xaxis_title="Time Period",
        yaxis_title="Total Revenue (£)",
        hovermode='x unified',
        showlegend=False,
        yaxis=dict(tickformat='£,.0f')
    )
    
    fig.update_traces(
        line=dict(color='#ff7f0e', width=3),
        marker=dict(size=8)
    )
    
    return fig

# Define main dashboard function.
def main():
    """Main dashboard function."""
    
    # Dashboard title
    st.title("📊 Enhanced Sales Analysis Dashboard with Simple Moving Average Forecasting")
    st.markdown("---")
    
    # Load data
    monthly_df, weekly_df = load_data()
    
    if monthly_df.empty and weekly_df.empty:
        st.error("No data available. Please ensure the CSV files are in the correct location.")
        return
    
    # Create product options
    product_options = create_product_options(monthly_df, weekly_df)
    
    if not product_options:
        st.error("No products found in the data.")
        return
    
    # Sidebar for controls
    st.sidebar.header("📋 Analysis Controls")
    
    # Create two columns for dropdowns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Select Product")
        selected_product_display = st.selectbox(
            "Choose a product:",
            options=list(product_options.keys()),
            key="product_selector"
        )
        selected_stockcode = product_options[selected_product_display]
    
    with col2:
        st.subheader("📅 Analysis Period")
        analysis_type = st.selectbox(
            "Choose analysis type:",
            options=['monthly', 'weekly'],
            index=0,  # Default to monthly
            key="period_selector"
        )
    
    # Forecasting controls in sidebar
    st.sidebar.header("🔮 SMA Forecasting Controls")
    enable_forecasting = st.sidebar.checkbox("Enable SMA Forecasting", value=True)
    
    if enable_forecasting:
        forecast_periods = st.sidebar.slider(
            "Forecast Periods:",
            min_value=1,
            max_value=12,
            value=3,
            help="Number of periods to forecast"
        )
        
        window_size = st.sidebar.slider(
            "Moving Average Window:",
            min_value=2,
            max_value=10,
            value=3,
            help="Number of periods to use for moving average"
        )
        
        enable_evaluation = st.sidebar.checkbox(
            "Enable Model Evaluation",
            value=True,
            help="Split data to evaluate forecast accuracy"
        )
    
    # Filter data based on selection
    if analysis_type == 'monthly':
        filtered_data = monthly_df[monthly_df['StockCode'] == selected_stockcode].copy()
        filtered_data = filtered_data.sort_values('YearMonth')
    else:
        filtered_data = weekly_df[weekly_df['StockCode'] == selected_stockcode].copy()
        filtered_data = filter_recent_weeks(filtered_data, weeks=10)
        filtered_data = filtered_data.sort_values('YearWeek')
    
    # Display selected product info
    if not filtered_data.empty:
        product_description = filtered_data['Description'].iloc[0]
        st.info(f"**Selected Product:** {selected_stockcode} - {product_description}")
    else:
        st.warning("No data available for the selected product and time period.")
        return
    
    # Create metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    if analysis_type == 'monthly':
        total_qty = filtered_data['Monthly_Quantity_Sold'].sum()
        total_revenue = filtered_data['Monthly_Total_Price'].sum()
        avg_qty = filtered_data['Monthly_Quantity_Sold'].mean()
        periods = len(filtered_data)
    else:
        total_qty = filtered_data['Weekly_Quantity_Sold'].sum()
        total_revenue = filtered_data['Weekly_Total_Price'].sum()
        avg_qty = filtered_data['Weekly_Quantity_Sold'].mean()
        periods = len(filtered_data)
    
    with col1:
        st.metric("Total Quantity Sold", f"{total_qty:,.0f} units")
    with col2:
        st.metric("Total Revenue", f"£{total_revenue:,.2f}")
    with col3:
        st.metric("Average per Period", f"{avg_qty:.1f} units")
    with col4:
        st.metric("Active Periods", f"{periods}")
    
    # SMA Forecasting Section
    if enable_forecasting:
        st.header("🔮 Simple Moving Average Forecasting Analysis")
        
        # Get time series data for forecasting
        if analysis_type == 'monthly':
            qty_series = filtered_data.set_index('YearMonth')['Monthly_Quantity_Sold']
            price_series = filtered_data.set_index('YearMonth')['Monthly_Per_Unit_Price']
            revenue_series = filtered_data.set_index('YearMonth')['Monthly_Total_Price']
        else:
            qty_series = filtered_data.set_index('YearWeek')['Weekly_Quantity_Sold']
            price_series = filtered_data.set_index('YearWeek')['Weekly_Per_Unit_Price']
            revenue_series = filtered_data.set_index('YearWeek')['Weekly_Total_Price']
        
        # Model Evaluation
        if enable_evaluation and len(filtered_data) > 5:
            st.subheader("📊 Model Evaluation Results")
            
            eval_col1, eval_col2, eval_col3 = st.columns(3)
            
            # Evaluate each metric
            for i, (series, series_name, col) in enumerate([
                (qty_series, "Quantity", eval_col1),
                (price_series, "Price", eval_col2),
                (revenue_series, "Revenue", eval_col3)
            ]):
                
                with col:
                    st.write(f"**{series_name} Forecasting**")
                    
                    # Split data
                    train_data, test_data = train_test_split_ts(series, test_ratio=0.3)
                    
                    if len(test_data) > 0 and len(train_data) >= window_size:
                        # Forecast on test period
                        test_forecast, _, _ = simple_moving_average_forecast(
                            train_data, window_size, len(test_data)
                        )
                        
                        # Calculate metrics
                        if test_forecast and len(test_forecast) == len(test_data):
                            metrics = calculate_evaluation_metrics(test_data.values, test_forecast)
                            
                            if metrics:
                                st.markdown('<div class="evaluation-metrics">', unsafe_allow_html=True)
                                st.metric("MAE", f"{metrics['MAE']:.2f}")
                                st.metric("RMSE", f"{metrics['RMSE']:.2f}")  
                                st.metric("MAPE", f"{metrics['MAPE']:.1f}%")
                                st.metric("R²", f"{metrics['R2']:.3f}")
                                st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.warning("Evaluation failed")
                    else:
                        st.warning("Insufficient data for evaluation")
        
        # Generate forecasts for full dataset
        st.subheader("🎯 Forecast Results")
        
        # Forecast for each metric
        qty_forecast, qty_model, qty_ci = simple_moving_average_forecast(
            qty_series, window_size, forecast_periods
        )
        price_forecast, price_model, price_ci = simple_moving_average_forecast(
            price_series, window_size, forecast_periods
        )
        revenue_forecast, revenue_model, revenue_ci = simple_moving_average_forecast(
            revenue_series, window_size, forecast_periods
        )
        
        # Display forecasts in tabs
        tab1, tab2, tab3 = st.tabs(["📦 Quantity Forecast", "💷 Price Forecast", "💰 Revenue Forecast"])
        
        with tab1:
            st.markdown('<div class="forecast-container">', unsafe_allow_html=True)
            
            # Forecast chart
            qty_fig = create_forecast_visualization(
                filtered_data, qty_forecast, qty_ci, analysis_type, 'quantity', 
                selected_product_display, window_size
            )
            st.plotly_chart(qty_fig, use_container_width=True)
            
            # Forecast values
            st.write(f"**Next {min(3, len(qty_forecast))} Period Quantity Forecasts (SMA-{window_size}):**")
            for i, val in enumerate(qty_forecast[:3], 1):
                if qty_ci and i <= len(qty_ci['lower']):
                    st.write(f"Period {i}: {val:.0f} units (95% CI: {qty_ci['lower'][i-1]:.0f} - {qty_ci['upper'][i-1]:.0f})")
                else:
                    st.write(f"Period {i}: {val:.0f} units")
            
            if qty_model:
                st.info(f"📊 Based on average of last {qty_model['window_size']} periods: {qty_model['forecast_value']:.0f} units")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<div class="forecast-container">', unsafe_allow_html=True)
            
            # Forecast chart
            price_fig = create_forecast_visualization(
                filtered_data, price_forecast, price_ci, analysis_type, 'price', 
                selected_product_display, window_size
            )
            st.plotly_chart(price_fig, use_container_width=True)
            
            # Forecast values
            st.write(f"**Next {min(3, len(price_forecast))} Period Price Forecasts (SMA-{window_size}):**")
            for i, val in enumerate(price_forecast[:3], 1):
                if price_ci and i <= len(price_ci['lower']):
                    st.write(f"Period {i}: £{val:.2f} (95% CI: £{price_ci['lower'][i-1]:.2f} - £{price_ci['upper'][i-1]:.2f})")
                else:
                    st.write(f"Period {i}: £{val:.2f}")
            
            if price_model:
                st.info(f"📊 Based on average of last {price_model['window_size']} periods: £{price_model['forecast_value']:.2f}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            st.markdown('<div class="forecast-container">', unsafe_allow_html=True)
            
            # Forecast chart
            revenue_fig = create_forecast_visualization(
                filtered_data, revenue_forecast, revenue_ci, analysis_type, 'revenue', 
                selected_product_display, window_size
            )
            st.plotly_chart(revenue_fig, use_container_width=True)
            
            # Forecast values
            st.write(f"**Next {min(3, len(revenue_forecast))} Period Revenue Forecasts (SMA-{window_size}):**")
            for i, val in enumerate(revenue_forecast[:3], 1):
                if revenue_ci and i <= len(revenue_ci['lower']):
                    st.write(f"Period {i}: £{val:.2f} (95% CI: £{revenue_ci['lower'][i-1]:.2f} - £{revenue_ci['upper'][i-1]:.2f})")
                else:
                    st.write(f"Period {i}: £{val:.2f}")
            
            if revenue_model:
                st.info(f"📊 Based on average of last {revenue_model['window_size']} periods: £{revenue_model['forecast_value']:.2f}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
    # Trend Analysis
    st.markdown("---")
    st.header("📊 Trend Analysis")
    
    with st.expander("🔬 Detailed Analysis"):
        try:
            if analysis_type == 'monthly':
                ts_qty = filtered_data.set_index('YearMonth')['Monthly_Quantity_Sold']
                ts_price = filtered_data.set_index('YearMonth')['Monthly_Per_Unit_Price']
            else:
                ts_qty = filtered_data.set_index('YearWeek')['Weekly_Quantity_Sold']
                ts_price = filtered_data.set_index('YearWeek')['Weekly_Per_Unit_Price']
            
            if len(ts_qty) >= 5:  # Minimum for trend analysis
                trend_col1, trend_col2 = st.columns(2)
                
                with trend_col1:
                    st.subheader("📈 Quantity Trend")
                    qty_trend = ts_qty.rolling(window=min(3, len(ts_qty)//2)).mean().iloc[-1]
                    qty_start = ts_qty.iloc[0]
                    qty_change = ((qty_trend - qty_start) / qty_start) * 100 if qty_start != 0 else 0
                    
                    if qty_change > 10:
                        st.success(f"📈 Strong upward trend (+{qty_change:.1f}%)")
                    elif qty_change > 0:
                        st.info(f"↗️ Mild upward trend (+{qty_change:.1f}%)")
                    elif qty_change < -10:
                        st.error(f"📉 Strong downward trend ({qty_change:.1f}%)")
                    elif qty_change < 0:
                        st.warning(f"↘️ Mild downward trend ({qty_change:.1f}%)")
                    else:
                        st.info("➡️ Stable trend")
                
                with trend_col2:
                    st.subheader("💷 Price Trend")
                    price_trend = ts_price.rolling(window=min(3, len(ts_price)//2)).mean().iloc[-1]
                    price_start = ts_price.iloc[0]
                    price_change = ((price_trend - price_start) / price_start) * 100 if price_start != 0 else 0
                    
                    if price_change > 10:
                        st.success(f"📈 Strong upward trend (+{price_change:.1f}%)")
                    elif price_change > 0:
                        st.info(f"↗️ Mild upward trend (+{price_change:.1f}%)")
                    elif price_change < -10:
                        st.error(f"📉 Strong downward trend ({price_change:.1f}%)")
                    elif price_change < 0:
                        st.warning(f"↘️ Mild downward trend ({price_change:.1f}%)")
                    else:
                        st.info("➡️ Price Stable")
            else:
                st.warning("Insufficient data for detailed trend analysis.")
        
        except Exception as e:
            st.warning(f"Advanced analysis not available: {e}")

    # Create charts
    st.markdown("---")
    
    # Row 1: Quantity Chart
    st.subheader("📈 Quantity Sold Analysis")
    quantity_chart = create_quantity_chart(filtered_data, analysis_type, selected_product_display)
    st.plotly_chart(quantity_chart, use_container_width=True)
    
    # Row 2: Per Unit Price Chart
    st.subheader("💷 Per Unit Price Analysis")
    per_unit_chart = create_per_unit_price_chart(filtered_data, analysis_type, selected_product_display)
    st.plotly_chart(per_unit_chart, use_container_width=True)
    
    # Row 3: Revenue Chart
    st.subheader("💰 Revenue Analysis")
    price_chart = create_price_chart(filtered_data, analysis_type, selected_product_display)
    st.plotly_chart(price_chart, use_container_width=True)
    
    # Data preview
    if st.checkbox("📋 Show Data Table"):
        st.subheader("Data Preview")
        st.dataframe(filtered_data, use_container_width=True)
    
    # Download section
    if not filtered_data.empty:
        st.markdown("---")
        st.subheader("💾 Download Data & Results")
        
        download_col1, download_col2 = st.columns(2)
        
        with download_col1:
            # Original data download
            csv = filtered_data.to_csv(index=False)
            filename = f"{selected_stockcode}_{analysis_type}_analysis.csv"
            
            st.download_button(
                label=f"📥 Download {analysis_type.title()} Data",
                data=csv,
                file_name=filename,
                mime='text/csv'
            )
        
        with download_col2:
            # Forecast results download
            if enable_forecasting:
                # Create forecast summary
                forecast_summary = []
                
                if analysis_type == 'monthly':
                    last_date = filtered_data['YearMonth'].max()
                    future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(forecast_periods)]
                else:
                    last_date = filtered_data['YearWeek'].max()
                    future_dates = [last_date + timedelta(weeks=i+1) for i in range(forecast_periods)]
                
                qty_forecast, _, _ = simple_moving_average_forecast(qty_series, window_size, forecast_periods)
                price_forecast, _, _ = simple_moving_average_forecast(price_series, window_size, forecast_periods)
                revenue_forecast, _, _ = simple_moving_average_forecast(revenue_series, window_size, forecast_periods)
                
                for i, date in enumerate(future_dates):
                    if i < len(qty_forecast) and i < len(price_forecast) and i < len(revenue_forecast):
                        forecast_summary.append({
                            'Date': date,
                            'Forecasted_Quantity': qty_forecast[i],
                            'Forecasted_Price': price_forecast[i],
                            'Forecasted_Revenue': revenue_forecast[i]
                        })
                
                if forecast_summary:
                    forecast_df = pd.DataFrame(forecast_summary)
                    forecast_csv = forecast_df.to_csv(index=False)
                    
                    st.download_button(
                        label="🔮 Download Forecasts",
                        data=forecast_csv,
                        file_name=f"{selected_stockcode}_{analysis_type}_forecasts.csv",
                        mime='text/csv'
                    )

# Define function to add information to sidebar.
def add_sidebar_info():
    """Add information to sidebar."""
    
    st.sidebar.header("ℹ️ Dashboard Info")
    
    st.sidebar.markdown("""
    **SMA Features:**
    - Simple Moving Average (SMA) forecasting
    - Model evaluation metrics (MAE, RMSE, MAPE)
    - Confidence intervals for forecasts
    - Rolling trend analysis
    
    **General Features:**
    - Select any product by stock code
    - Switch between monthly/weekly views
    - Weekly view shows recent 10 weeks
    - Interactive charts with hover details
    - Download filtered data and forecasts as CSV
    
    **Data Requirements:**
    - monthly_sales_aggregated.csv
    - weekly_sales_aggregated.csv
    
    **Chart Details:**
    - Blue line: Historical data
    - Green dotted: Historical SMA
    - Orange dashed: SMA forecasts
    - Shaded area: 95% confidence intervals
    """)

if __name__ == "__main__":
    add_sidebar_info()
    main()