import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Utility Functions
def get_stock_data(ticker_symbol, period="1y"):
    ticker = yf.Ticker(ticker_symbol)
    data = ticker.history(period=period)
    info = ticker.info
    return data, info

def generate_scenarios(data, percentage_change=10):
    """Simulate bullish and bearish scenarios."""
    bullish = data['Close'] * (1 + percentage_change / 100)
    bearish = data['Close'] * (1 - percentage_change / 100)
    return bullish, bearish

def predict_future_prices(data, days=30):
    """Simple predictive model for future stock prices."""
    data = data.reset_index()
    data['Date_Ordinal'] = pd.to_datetime(data['Date']).map(pd.Timestamp.toordinal)
    X = data[['Date_Ordinal']]
    y = data['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    future_dates = [data['Date_Ordinal'].max() + i for i in range(1, days + 1)]
    future_prices = model.predict(np.array(future_dates).reshape(-1, 1))
    future_df = pd.DataFrame({'Date': pd.to_datetime(future_dates, origin='unix', unit='D'), 'Predicted Price': future_prices})
    return future_df

# Streamlit Pages
def home():
    st.title("Stock Valuation Dashboard")
    st.write("Analyze individual stocks or create a portfolio.")
    st.write("Navigate using the sidebar.")
    st.write("Data powered by Yahoo Finance and advanced analytics.")

def stock_overview():
    st.title("Stock Overview")
    stock = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA):", "AAPL")
    period = st.selectbox("Select Time Period", ["1mo", "3mo", "6mo", "1y", "5y", "max"])
    if st.button("Analyze Stock"):
        data, info = get_stock_data(stock, period)
        st.subheader(f"{info['shortName']} ({stock})")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
        st.plotly_chart(fig)
        st.write("Key Metrics")
        st.json({k: info[k] for k in ['sector', 'industry', 'marketCap', 'forwardPE', 'dividendYield'] if k in info})

def scenario_analysis():
    st.title("Scenario Analysis")
    stock = st.text_input("Enter Stock Ticker for Scenario Analysis:", "AAPL")
    percentage_change = st.slider("Select Scenario Percentage Change (%)", min_value=1, max_value=50, value=10)
    if st.button("Run Scenarios"):
        data, _ = get_stock_data(stock, "1y")
        bullish, bearish = generate_scenarios(data, percentage_change)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=data.index, y=bullish, mode='lines', name='Bullish Scenario'))
        fig.add_trace(go.Scatter(x=data.index, y=bearish, mode='lines', name='Bearish Scenario'))
        st.plotly_chart(fig)

def ai_insights():
    st.title("AI-Powered Insights")
    stock = st.text_input("Enter Stock Ticker for AI Predictions:", "AAPL")
    days = st.slider("Predict Days Ahead:", min_value=1, max_value=60, value=30)
    if st.button("Predict Future Prices"):
        data, _ = get_stock_data(stock, "1y")
        predictions = predict_future_prices(data, days)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical Prices'))
        fig.add_trace(go.Scatter(x=predictions['Date'], y=predictions['Predicted Price'], mode='lines', name='Predicted Prices'))
        st.plotly_chart(fig)
        st.write("Future Price Predictions")
        st.dataframe(predictions)

def gamification():
    st.title("Learn Valuation Principles")
    st.write("Test your understanding of valuation metrics and portfolio management.")
    question = st.selectbox("Which valuation method calculates the present value of future cash flows?",
                             options=["P/E Ratio", "DCF (Discounted Cash Flow)", "Net Asset Value"])
    if st.button("Submit Answer"):
        if question == "DCF (Discounted Cash Flow)":
            st.success("Correct! The DCF method calculates the present value of future cash flows.")
        else:
            st.error("Incorrect. The correct answer is DCF (Discounted Cash Flow).")

# Multi-Page Setup
PAGES = {
    "Home": home,
    "Stock Overview": stock_overview,
    "Scenario Analysis": scenario_analysis,
    "AI Insights": ai_insights,
    "Learn Valuation (Gamification)": gamification,
}

def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    page = PAGES[selection]
    page()

if __name__ == "__main__":
    main()
