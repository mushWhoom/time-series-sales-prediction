# time-series-sales-prediction
# E-Commerce Sales Time Series Forecasting

## Project Description
This project focuses on conducting Time Series Analysis and forecasting sales trends from an e-commerce platform. It processes raw sales transaction history across multiple months, combining them into a single comprehensive dataset to analyze underlying seasonal patterns and sales trends. 

The approach utilized ranges from traditional statistical modeling to advanced Deep Learning architectures to evaluate and compare their performance in predicting future sales.

## Dataset
The dataset consists of e-commerce transaction history with the following key attributes:
*   `Order ID`: The unique identification number for each order.
*   `Product`: The name of the item sold.
*   `Quantity Ordered`: The number of items purchased per order.
*   `Price Each`: The unit price of the item.
*   `Order Date`: The date and time of the transaction (The primary feature for Time Series analysis).
*   `Purchase Address`: The physical address where the order was placed.

## 🛠️ Methodology & Workflow
1.  **Data Preprocessing:** Merging multiple CSV files, cleaning the dataset by dropping null (NaN) values and duplicated headers, and converting the string date column into proper `datetime` format.
2.  **Exploratory Data Analysis (EDA):** Utilizing Seasonal Decomposition to break down the time series data into its Trend, Seasonality, and Residual (Noise) components.
3.  **Stationarity Testing:** Applying the Augmented Dickey-Fuller (ADF) Test to check if the data remains stable over time, a prerequisite for statistical modeling.
4.  **Statistical Modeling:** Implementing **ARIMA** (AutoRegressive Integrated Moving Average) for linear regression modeling of the time series.
5.  **Deep Learning Modeling:** Deploying **LSTM** (Long Short-Term Memory), **Bi-LSTM**, and **GRU** (Gated Recurrent Unit) architectures via Keras/TensorFlow to capture complex and non-linear time series patterns.
6.  **Model Evaluation:** Comparing model performances using metrics such as Mean Squared Error (MSE) and R-Squared (R2 Score).

---

## Data Analysis and Interpretation

Based on the data processing and modeling workflow performed in this notebook, here are the main interpretations drawn from the project:

### 1. Sales Data Characteristics (Trend & Seasonality)
Through the `seasonal_decompose` function and autocorrelation plots, e-commerce sales data generally exhibits **strong seasonal patterns**. Order spikes frequently align with specific periods (such as end-of-year or holiday seasons) and also display cyclic daily/weekly patterns driven by consumer shopping behaviors.

### 2. Data Stationarity (ADF Test)
Raw sales data is typically *non-stationary* (possessing a shifting trend and varying variance). For models like ARIMA to function optimally, the sales data must undergo a differencing phase to achieve stationarity (ensuring fluctuations hover around a constant mean).

### 3. Model Performance Comparison (ARIMA vs LSTM/GRU)
*   **ARIMA Model:** Highly effective at understanding the baseline movement of short-term time series statistically. However, this model struggles when presented with e-commerce data featuring extreme outliers or complex, multiple non-linear seasonalities.
*   **RNN/LSTM/GRU Models:** By leveraging `TimeseriesGenerator` and `MinMaxScaler`, Deep Learning architectures demonstrate the ability to retain long-term sales patterns (distant historical dependencies) while mitigating the vanishing gradient problem. The integration of *Dropout* and *BatchNormalization* layers successfully prevents the models from overfitting to temporary data spikes.

### 4. Forecasting Conclusion
Overall, Deep Learning models (such as LSTM and GRU) offer superior adaptability in forecasting e-commerce sales compared to conventional statistical models. The higher **R2 Score** and lower **MSE** achieved by the Deep Learning approaches indicate that stakeholders can rely on these systems for future inventory planning and optimizing marketing budget allocations in the upcoming months.

## How to Run the Project
1. Clone this repository: `git clone https://github.com/your-username/repo-name.git`
2. Ensure the required libraries are installed: `pip install pandas numpy matplotlib seaborn statsmodels scikit-learn tensorflow keras`
3. Execute the Jupyter Notebook / Google Colab file `Time-Series-Analysis.ipynb`
