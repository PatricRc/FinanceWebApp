import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import openai
import os  # Import for environment variables
from dotenv import load_dotenv  # Import dotenv to load .env file

# Load environment variables from .env file
load_dotenv()

# Utility Functions
def get_asset_data(ticker, asset_type, period="1y"):
    """Fetch data for stocks, ETFs, or cryptos."""
    if asset_type == "Crypto":
        ticker = f"{ticker}-USD"  # Format for Yahoo Finance crypto tickers
    data = yf.Ticker(ticker).history(period=period)
    return data

def calculate_portfolio_performance(portfolio):
    """Calculate weighted portfolio performance."""
    portfolio_data = pd.DataFrame()
    for asset, details in portfolio.items():
        weight = details['weight']
        data = get_asset_data(asset, details['type'], period="1y")
        data['Weighted Close'] = data['Close'] * weight
        portfolio_data[asset] = data['Weighted Close']
    portfolio_data['Portfolio Value'] = portfolio_data.sum(axis=1)
    return portfolio_data

def chat_with_portfolio(portfolio, input_text, api_key):
    """Chat with the user's portfolio data using OpenAI."""
    try:
        # Convert portfolio to a DataFrame for context
        portfolio_df = pd.DataFrame(portfolio).T
        portfolio_df = portfolio_df.reset_index().rename(columns={"index": "Asset"})
        context = portfolio_df.to_string(index=False)

        # Create a prompt template
        message = f"""
        Answer the following question using the context provided:

        Context:
        {context}

        Question:
        {input_text}

        Answer:
        """

        # Initialize OpenAI LLM with model 'gpt-4'
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a portfolio analyst."},
                      {"role": "user", "content": message}]
        )

        # Extract response
        st.write(response['choices'][0]['message']['content'].strip())

    except openai.error.OpenAIError as e:
        st.error(f"Error with OpenAI API: {e}")

# Streamlit Pages
def home():
    st.title("Dynamic Portfolio Management App")
    st.write("Build and track your portfolio with stocks, ETFs, and cryptocurrencies.")
    st.write("Navigate using the sidebar.")

def portfolio_management():
    st.title("Portfolio Management")
    
    # Input for assets
    st.subheader("Add Assets to Your Portfolio")
    asset_ticker = st.text_input("Enter Asset Ticker (e.g., AAPL, SPY, BTC):")
    asset_type = st.selectbox("Select Asset Type", ["Stock", "ETF", "Crypto"])
    asset_weight = st.slider("Weight in Portfolio (%)", min_value=0, max_value=100, value=0)
    
    # Portfolio storage
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = {}
    
    if st.button("Add Asset"):
        if asset_ticker and asset_weight > 0:
            if asset_ticker.upper() not in st.session_state.portfolio:
                st.session_state.portfolio[asset_ticker.upper()] = {
                    "type": asset_type,
                    "weight": asset_weight / 100
                }
                st.success(f"Added {asset_ticker.upper()} as a {asset_type} with {asset_weight}% weight.")
            else:
                st.error(f"Asset {asset_ticker.upper()} is already in your portfolio. Please edit or remove it first.")
        else:
            st.error("Please enter a valid ticker and weight.")
    
    # Remove Asset
    st.subheader("Remove Assets from Your Portfolio")
    if st.session_state.portfolio:
        remove_asset = st.selectbox("Select an Asset to Remove", list(st.session_state.portfolio.keys()))
        if st.button("Remove Asset"):
            del st.session_state.portfolio[remove_asset]
            st.success(f"Removed {remove_asset} from your portfolio.")
    
    # Show Portfolio
    st.subheader("Your Portfolio")
    if st.session_state.portfolio:
        portfolio_df = pd.DataFrame(st.session_state.portfolio).T
        portfolio_df["Weight (%)"] = portfolio_df["weight"] * 100
        st.dataframe(portfolio_df[["type", "Weight (%)"]])
        
        # Portfolio performance
        st.subheader("Portfolio Performance")
        portfolio_data = calculate_portfolio_performance(st.session_state.portfolio)
        st.line_chart(portfolio_data['Portfolio Value'])
    else:
        st.info("No assets in the portfolio yet. Add assets to see the performance.")

def asset_insights():
    st.title("Asset Insights")
    st.write("Get insights for individual assets in your portfolio.")
    
    # Select asset from the portfolio
    if "portfolio" in st.session_state and st.session_state.portfolio:
        asset = st.selectbox("Select an Asset", list(st.session_state.portfolio.keys()))
        if asset:
            asset_details = st.session_state.portfolio[asset]
            data = get_asset_data(asset, asset_details['type'], period="1y")
            
            # Display chart
            st.subheader(f"Performance of {asset} ({asset_details['type']})")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
            st.plotly_chart(fig)
            
            # Display key metrics
            st.write("Key Metrics")
            st.write(f"- Asset Type: {asset_details['type']}")
            st.write(f"- Latest Price: ${data['Close'][-1]:.2f}")
    else:
        st.info("No assets in the portfolio to analyze.")

def chat_with_portfolio_page():
    st.title("💬 Chat with Your Portfolio")

    if "portfolio" not in st.session_state or not st.session_state.portfolio:
        st.warning("Your portfolio is empty. Add assets to your portfolio first.")
        return

    # Display portfolio for context
    st.write("Your current portfolio:")
    portfolio_df = pd.DataFrame(st.session_state.portfolio).T
    portfolio_df = portfolio_df.reset_index().rename(columns={"index": "Asset"})
    st.dataframe(portfolio_df)

    # Text input for OpenAI API Key
    api_key = st.text_input("Enter your OpenAI API Key", type="password")

    # Enter the query for analysis
    input_text = st.text_area("Enter your query")

    # Perform analysis
    if input_text and api_key and st.button("Chat with Portfolio"):
        chat_with_portfolio(st.session_state.portfolio, input_text, api_key)

# Multi-Page Setup
PAGES = {
    "Home": home,
    "Portfolio Management": portfolio_management,
    "Asset Insights": asset_insights,
    "Chat with Portfolio": chat_with_portfolio_page,
}

def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    page = PAGES[selection]
    page()

if __name__ == "__main__":
    main()
