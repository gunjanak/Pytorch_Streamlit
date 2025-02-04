
import streamlit as st
import datetime
import plotly.express as px
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os




from dataframe_nepse import stock_dataFrame
# from forecast import forecast_closing_price

from basic import read_prepare_data, train_and_test,just_test


st.set_page_config(layout="wide")

st.write("Hello")
# Input from the user
user_input = st.text_input("Enter a symbol:")
# Date input from the user
selected_date = st.date_input(
    "Select a date:",
    value='2020-01-01'  # Default value (current date)
)


# Display the entered word
if st.button("Submit"):
    print("submit clicked")
    st.write("You entered:", user_input)
    st.write("Date :",selected_date)
    try:
        df = stock_dataFrame(user_input,selected_date)
        # List of all indicators
        df.ta.indicators()
    except Exception as e:
        print(f"{e}:occured")
        
    st.write(df.head())
    # columns = df.columns
    # st.write(columns)
    # Create a dropdown with column names
    selected_column = st.selectbox("Select a column:", list(df.columns))

    # Display the selected column
    st.write("You selected the column:", selected_column)
    
    if selected_column:
        fig = px.line(df, x=df.index, y=selected_column, title=f"{selected_column} Plot")
        st.plotly_chart(fig)
        
    technical_indicators = ['CCI','OBV','RSI','ADX',
                            'ENTROPY',
                            'VOLATILITY']
    # Streamlit app
    st.title("Technical Indicator Visualization")
    
    # Dropdown for selecting a technical indicator
    selected_indicator = st.selectbox("Select a Technical Indicator", technical_indicators)
    
    # Compute the selected technical indicator
    if selected_indicator == 'OBV':
        df["OBV"] = ta.obv(df["Close"], df["Volume"])
        indicator_data = df["OBV"]
        indicator_name = "On-Balance Volume (OBV)"
    elif selected_indicator == 'CCI':
        df["CCI"] = ta.cci(high=df["High"], low=df["Low"], close=df["Close"], length=20)
        indicator_data = df["CCI"]
        indicator_name = "Commodity Channel Index (CCI)"
    elif selected_indicator == 'RSI':
        df["RSI"] = ta.rsi(df["Close"], length=14)
        indicator_data = df["RSI"]
        indicator_name = "Relative Strength Index (RSI)"
    elif selected_indicator == "ADX":
        adx = ta.adx(df["High"], df["Low"], df["Close"], length=14)
        df["ADX"] = adx["ADX_14"]
        df["+DM"] = adx["DMP_14"]
        df["-DM"] = adx["DMN_14"]
        indicator_data = df[["ADX", "+DM", "-DM"]]
        indicator_name = "Average Directional Movement Index (ADX)"
    elif selected_indicator == 'ENTROPY':
        df["Entropy"] = ta.entropy(df["Close"], length=14)
        indicator_data = df["Entropy"]
        indicator_name = "Entropy"
    elif selected_indicator == 'VOLATILITY':
        df["Volatility"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
        indicator_data = df["Volatility"]
        indicator_name = "Volatility (ATR)"
    else:
        st.error("Invalid indicator selected")
        st.stop()

    # Display the dataframe with the selected indicator
    st.write(f"Data with {indicator_name}:")
    st.write(df.head())

    # Create candlestick chart
    candlestick = go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Candlestick"
    )

    # Create plots based on the selected indicator
    if selected_indicator == "ADX":
        adx_line = go.Scatter(
            x=df.index,
            y=df["ADX"],
            mode="lines",
            name="ADX",
            line=dict(color="orange")
        )
        dmp_line = go.Scatter(
            x=df.index,
            y=df["+DM"],
            mode="lines",
            name="Positive Directional Movement (+DM)",
            line=dict(color="green")
        )
        dmn_line = go.Scatter(
            x=df.index,
            y=df["-DM"],
            mode="lines",
            name="Negative Directional Movement (-DM)",
            line=dict(color="red")
        )
    else:
        indicator_line = go.Scatter(
            x=df.index,
            y=indicator_data,
            mode="lines",
            name=indicator_name,
            line=dict(color="orange")
        )

    # Combine the plots in a subplot
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Candlestick Chart", indicator_name)
    )

    # Add plots to subplots
    fig.add_trace(candlestick, row=1, col=1)
    if selected_indicator == "ADX":
        fig.add_trace(adx_line, row=2, col=1)
        fig.add_trace(dmp_line, row=2, col=1)
        fig.add_trace(dmn_line, row=2, col=1)
    else:
        fig.add_trace(indicator_line, row=2, col=1)

    # Update layout for better visualization
    fig.update_layout(
        height=600,
        width=1200,
        xaxis_rangeslider_visible=False,
        title=f"Candlestick Chart with {indicator_name}",
        showlegend=True
    )

    # Display the figure
    st.plotly_chart(fig)

    
    
    
    try:
        df,model_path = read_prepare_data(user_input=user_input)
        test_mape = just_test(df,model_path=model_path)
        st.write(f"Test ERROR:{test_mape}")
        while test_mape > 100:
            model_path = f"{user_input}_price_forecaster_gru_l2_four_layers.pth"
            mape,test_mape = train_and_test(df,model_path=model_path)
            st.write(f"Train ERROR: {mape}, Test ERROR:{test_mape}")
        # Streamlit app
        # st.title("Stock Data Visualization")
        # print(df_test.tail())

        # # Plotting the data
        # fig = px.line(df_test, x=df_test.index, y=["Close", "Predicted Value"], markers=True,
        #             labels={"value": "Price", "variable": "Type"},
        #             title="Close and Predicted Value Over Time")

        # # Display the plot
        # st.plotly_chart(fig)
    except Exception as e:
        print(e)
        

 