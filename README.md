# Online Retail Sales Analysis - Prakhar Gupta

This sales analysis project uses a dataset of online retail transactions to perform customer segmentation and sales forecasting. It includes several Python scripts and notebooks that clean, process, and analyze the data to provide insights into customer behavior and predict future sales trends.

### Key Features and Functionality

* **Data Processing and Cleaning** 🧹: The project starts by cleaning the raw retail data. This includes handling missing values, identifying and managing canceled orders, and creating new features like `TotalAmount`.
* **Customer Segmentation** 🎯: The project uses RFMC (Recency, Frequency, Monetary, Cancellation Rate) analysis to segment customers into different groups such as "Champions," "Loyal Regulars," "At Risk," and "Inactive" customers. It employs various clustering algorithms like K-Means, Hierarchical Clustering, and Gaussian Mixture Models (GMM) to achieve this.
* **Sales Forecasting** 📈: The project provides sales forecasting using a simple moving average (SMA) model. It allows users to forecast future sales based on historical data and evaluate the model's performance using metrics like MAE, RMSE, and MAPE.
* **Interactive Dashboard** 📊: The project includes an interactive dashboard built with Streamlit that allows users to explore the data, analyze customer segments, and view sales forecasts for different products.
* **Chatbot Integration** 🤖: A chatbot feature is integrated into the dashboard, enabling users to ask questions about the data and receive analysis code in response.

### How to Get Started

1.  **Set up your environment**: Ensure you have Python and the required libraries installed.
2.  **Run the Streamlit app**: Execute the `sales.py` script to launch the main dashboard.
3.  **Explore the dashboard**: Use the sidebar to navigate between different pages, including the customer segmentation analysis and sales forecast modules.
4.  **Use the chatbot**: Ask questions about the data using the chatbot feature to get instant insights.

### Code Structure

* `sales.py`: The main Streamlit application file for data processing and EDA.
* `pages/cluster.py`: Contains the code for the customer segmentation dashboard.
* `pages/sales_forecast.py`: Includes the sales forecasting module with SMA.
* `pages/chat.py`: Implements the chatbot functionality.
* `experiment_notebooks/`: Contains Jupyter notebooks for data processing and customer segmentation experiments.
