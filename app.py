import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import seaborn as sns

import plotly.express as px
import streamlit as st
import pandas_datareader as pdr
from datetime import datetime

import streamlit as st


import yfinance as yf


st.title('Stock Trend Analysis')


ticker = st.sidebar.text_input('Enter Stock Sticker','AAPL')
start_date = st.sidebar.date_input('Start Date')
end_date = st.sidebar.date_input('End Date')


df = yf.download(ticker, start=start_date, end=end_date)
st.subheader('Data for given data')
st.write(df.describe())

# Plot 1 the behaviour
sns.set_style('whitegrid')
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['Close'], color='green', linewidth=3, label='Closing Price')
ax.set_title('Closing Price for given dates')
ax.set_xlabel('Date')
ax.set_ylabel('Closing Price')
ax.legend()
st.pyplot(fig)

# Plot 2 moving average over 100 days
# st.set_option('deprecation.showPyplotGlobalUse', False)
# n =  st.slider("Enter MA for how many days ", 0, 3000, 100)
# ma = df['Close'].rolling(n).mean()
# plt.figure(figsize=(12, 6))
# # Convert the index to a numpy array
# dates = df.index.to_numpy()
# # Convert the 'Close' values to a numpy array
# close_values = df['Close'].to_numpy()
# plt.plot(dates, close_values, color='blue', linewidth=2, label='Closing Price')
# # Convert the 100-day Moving Average to a DataFrame
# ma_df = ma.to_frame()
# # Convert the index and values of the Moving Average DataFrame to numpy arrays
# ma_dates = ma_df.index.to_numpy()
# ma_values = ma_df.iloc[:, 0].to_numpy()
# plt.plot(ma_dates, ma_values, color='red', linewidth=2, label='Moving Average (100 days)')
# plt.title('Stock Closing Prices with Moving Average ', fontsize=16)
# plt.xlabel('Date', fontsize=14)
# plt.ylabel('Closing Price', fontsize=14)
# plt.grid(True, linestyle='--', linewidth=0.5)
# plt.gcf().autofmt_xdate()
# plt.legend(loc='upper left', fontsize=12)
# plt.tight_layout()
# st.pyplot()


pricing_data, fundamental_data, news, compare_stocks= st.tabs(["Pricing Data", "Fundamental Data", "Top Io News","Compare Stocks"])

with pricing_data:
    st.header('Price Movements')
    data2 = df
    data2['% Change'] = df['Adj Close']/ df['Adj Close'].shift (1) - 1
    data2.dropna (inplace=True)
    st.write(data2)
    annual_return = data2['% Change'].mean() * 252 * 100
    st.write('Annual Return is ', annual_return, '%')
    stdev = np.std(data2['% Change'])*np.sqrt(252)
    st.write('Standard Deviation is ',stdev*100, '%')
    st.write('Risk Adj. Return is ', annual_return/(stdev*100))
    st.set_option('deprecation.showPyplotGlobalUse', False)
    n =  st.slider("Enter MA for how many days ", 0, 3000, 100)
    ma = df['Close'].rolling(n).mean()
    plt.figure(figsize=(12, 6))
    # Convert the index to a numpy array
    dates = df.index.to_numpy()
    # Convert the 'Close' values to a numpy array
    close_values = df['Close'].to_numpy()
    plt.plot(dates, close_values, color='blue', linewidth=2, label='Closing Price')
    # Convert the 100-day Moving Average to a DataFrame
    ma_df = ma.to_frame()
    # Convert the index and values of the Moving Average DataFrame to numpy arrays
    ma_dates = ma_df.index.to_numpy()
    ma_values = ma_df.iloc[:, 0].to_numpy()
    plt.plot(ma_dates, ma_values, color='red', linewidth=2, label='Moving Average (100 days)')
    plt.title('Stock Closing Prices with Moving Average ', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Closing Price', fontsize=14)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.gcf().autofmt_xdate()
    plt.legend(loc='upper left', fontsize=12)
    plt.tight_layout()
    st.pyplot()



from alpha_vantage.fundamentaldata import FundamentalData 
with fundamental_data:
        
# Check if the user entered a ticker symbol
        if ticker:
            # Retrieve fundamental data using yfinance
            stock_data = yf.Ticker(ticker)

            # Get and display the balance sheet data
            st.subheader('Balance Sheet')
            balance_sheet = stock_data.balance_sheet
            st.write(balance_sheet)

            # Get and display the income statement data
            st.subheader('Income Statement')
            income_statement = stock_data.financials
            st.write(income_statement)

            # Get and display the cash flow statement data
            st.subheader('Cash Flow Statement')
            cash_flow = stock_data.cashflow
            st.write(cash_flow)


import streamlit as st
from stocknews import StockNews
with news:
        if ticker:
            sn = StockNews(ticker, save_news=False)
            df_news = sn.read_rss()
            st.header(f'News of {ticker}')
            for i in range(10):
                st.subheader(f'News {i+1}')
                st.write(df_news['published'][i])
                st.write(df_news['title'][i])
                st.write(df_news['summary'][i])
                title_sentiment = df_news['sentiment_title'][i]
                st.write(f'Title Sentiment: {title_sentiment}')
                news_sentiment = df_news['sentiment_summary'][i]
                st.write(f'News Sentiment: {news_sentiment}')


with compare_stocks:
     
     #Comparison of two stocks
   
# Function to retrieve historical stock data and plot comparison
        def plot_stock_comparison(stock1, stock2, start_date, end_date):
            try:
                # Retrieve the historical stock data for stock1 and stock2
                df_stock1 = yf.download(stock1, start=start_date, end=end_date)
                df_stock2 = yf.download(stock2, start=start_date, end=end_date)

                # Plot the closing prices of both stocks on the same graph
                plt.figure(figsize=(12, 6))

                plt.plot(df_stock1.index.to_numpy(), df_stock1['Close'].values, label=stock1, color='blue', linewidth=2)
                plt.plot(df_stock2.index.to_numpy(), df_stock2['Close'].values, label=stock2, color='green', linewidth=2)

                plt.title(f'Comparison of {stock1} and {stock2} Closing Prices', fontsize=16)
                plt.xlabel('Date', fontsize=14)
                plt.ylabel('Closing Price', fontsize=14)
                plt.grid(True, linestyle='--', linewidth=0.5)
                plt.gcf().autofmt_xdate()
                plt.legend(loc='upper left', fontsize=12)

                plt.tight_layout()

                # Display the plot
                st.pyplot()

            except Exception as e:
                st.error(f"An error occurred: {e}")

        # Take input from the user using Streamlit widgets
        st.title("Stock Price Comparison")
        stock_symbol1 = st.text_input("Enter the stock symbol for company 1:")
        stock_symbol2 = st.text_input("Enter the stock symbol for company 2:")
        start_date = st.date_input("Enter the start date:")
        end_date = st.date_input("Enter the end date:")

# Create a button to trigger the plot
if st.button("Plot Comparison"):
    # Validate inputs
    if stock_symbol1 and stock_symbol2 and start_date < end_date:
        # Call the function to plot the comparison
        plot_stock_comparison(stock_symbol1, stock_symbol2, start_date, end_date)
    else:
        st.warning("Please provide valid inputs: stock symbols and a valid date range.")



