import streamlit as st
import pandas as pd
from stock import stock_dataFrame
from helper import prepare_data,just_test,train_and_test

# Title of the app
st.title("Stock Price Forecast App ")

# Text input
user_input = st.text_input("Enter some text:")

# Display the input
df = pd.DataFrame({})
try:
    if st.button("Get the data"):
        df = stock_dataFrame(stock_symbol=user_input)
except Exception as e:
    st.write(e)
    
st.write(df.head()) 
df,model_path = prepare_data(df,user_input)
print(df.head()) 
print(model_path)
mape,test_mape = train_and_test(df,model_path)
st.write(mape)
st.write(test_mape)