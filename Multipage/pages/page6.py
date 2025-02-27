import streamlit as st
import pandas as pd
from stock import stock_dataFrame
from helper import prepare_data,just_test,train_and_test

# Title of the app
st.title("Stock Price Forecast Appplication ")

# Text input
user_input = st.text_input("Enter some text:")
weekly = st.checkbox("Do you need weekly prediction?")
print(weekly)

# Display the input
df = pd.DataFrame({})
try:
    if st.button("Get the data"):
        if weekly:
            df = stock_dataFrame(stock_symbol=user_input,weekly=True)
        else:
            df = stock_dataFrame(stock_symbol=user_input)
except Exception as e:
    st.write(e)
    
st.write(df.head()) 
df,model_path = prepare_data(df,user_input)
print(df.head()) 
print(model_path)
mape,test_mape,df_test,predicted_price = train_and_test(df,model_path)
st.write(mape)
st.write(test_mape)
st.write(df_test.tail(10))
st.write(predicted_price)