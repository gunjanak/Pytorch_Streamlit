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
from forecast_bgru import train_gru_model,test_gru_model
from helper import normalize_with_sklearn,denormalize_with_sklearn

def read_prepare_data(user_input,selected_date = "2020-01-01"):
    try:
        df = stock_dataFrame(user_input,selected_date)
        # List of all indicators
        # df.ta.indicators()
    except Exception as e:
        print(f"{e}:occured")
    
    
    #Calculate RSI  
    df["RSI"] = ta.rsi(df["Close"], length=14)
    df = df[['Close','RSI']]

    df = df.dropna()
    # check if the model for particular symbol exist
    model_path = f"{user_input}_price_forecaster_gru.pth"
    print(model_path)
    if os.path.exists(model_path):
        print("File exists")
    else:
        print("File does not exist")
        model_path = "price_forecaster_gru.pth"
        
    
    print(df.head())
    
    return df,model_path

   

    
def train_and_test(df,model_path):
    # model_path= "rsi_bgru.pth"
     #Train and test split
    train_size = 0.8
    split_index = int(len(df) * train_size)
    df_train = df[:split_index]
    df_test = df[split_index:]


    #Normalize
    columns_to_normalize = ['Close', 'RSI']
    df_train_normalized, scaler_train = normalize_with_sklearn(df_train, 
                                                               columns_to_normalize)
    df_test_normalized, scaler_test = normalize_with_sklearn(df_test,
                                                             columns_to_normalize)

    
    #Train
    n_days = 5


    model, mape = train_gru_model(df_train_normalized, n_days,model_path=model_path,
                                epochs=2000)
    print(len(df_train_normalized))


    #Predict on test data
    predictions,test_mape = test_gru_model(model,df_test_normalized,n_days)
    print(len(predictions))
    print(predictions)

    # Re-transform predictions to original scale
    # Denormalize predictions to original scale (for 'Close' column only)
    predictions_original_scale = denormalize_with_sklearn(predictions,scaler_test, column_index=0)

    print(f"Testing MAPE (Original Scale): {test_mape:.2f}%")

    print("predictions_original_scale")
    print(predictions_original_scale)
    # # # Save the trained model

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    # # # # Convert predictions to a Pandas Series with the appropriate index
    predictions_series = pd.Series(predictions_original_scale, index=df_test.index[n_days:])
    print(predictions_series)
    print(df_test)

    # # # Rename the series to a meaningful column name, e.g., 'Predicted Value'
    predictions_series = predictions_series.rename('Predicted Value')

    # # # Merge the series with the DataFrame on the index
    merged_df = df_test.merge(predictions_series, left_index=True,
                            right_index=True, how='left')

    # # print(merged_df)
    df_test = merged_df.dropna()



    # # # # Print the last few rows to verify
    print(df_test.tail())


    
    
    return mape,test_mape


#read the data from api
#symbol = "ADBL","NTC","JOSHI","KBL","MMKJL","NICA","PHCL",
# "OHL","SCBD","SHLB","SINDU","UNL","WNLB"
user_input = "ANLB"
# selected_date = "2020-01-01"


df,model_path = read_prepare_data(user_input=user_input)
mape,test_mape = train_and_test(df,model_path=model_path)
print(f"Train ERROR: {mape}, Test ERROR:{test_mape}")

#retrain the model if test_mape is greater than 100
while test_mape>100:
    model_path = f"{user_input}_price_forecaster_gru.pth"
    mape,test_mape = train_and_test(df,model_path=model_path)
    print(f"Train ERROR: {mape}, Test ERROR:{test_mape}")
    
    
    