import streamlit as st
import pandas as pd
# from .. utlis import load_data,read_data
from utlis import (load_data,read_data,plot_histogram,plot_box,
                   plot_bar,plot_pie,plot_kde_chart,one_hot_encode,
                   train_and_evaluate_svm,train_and_evaluate_rf)



data_cleaned = False
st.title("Page 2")
st.write("This is the second additional page!")
uploaded_file,status = load_data()
if(status==True):
    df = read_data(uploaded_file)
    st.write(df.sample())
    st.write(df.columns)
    
    st.subheader("Choose a column to select as index")
    col_list = list(df.columns)
    selected_column = st.selectbox("Set column as index: ",col_list)
    
    st.write(f"You selected : {selected_column}")
   
    df = df.set_index(selected_column)
    st.write(df.sample())
    
        
    st.subheader("Describe the data")
    try:
        st.write(df.describe())
        
    except Exception as e:
        st.write(e)
        
    try:
        st.write(df.describe(include=['object','bool']))
    except Exception as e:
        st.write(e)
    
    
    #trying to clean the data
    st.write(df.isnull().sum())
    #get list of columns where isnull() is non zero
    missing_cols = df.columns[df.isnull().sum() > 0].tolist()
    st.write(missing_cols)
    handling_options = {}
    for col in missing_cols:
        col_type = df[col].dtype
        options = ["Drop Column","Fill with Mean","Fill with Median",
                   "Fill with Mode","Fill with Custom Value"]
        
        #allow user to select method
        choice = st.selectbox(f"How to handle missing values in **{col}**?",
                              options,key=col)
        handling_options[col] = choice
        
        #If user selectes fill with custom value , allow input
        if choice == "Fill with Custom Value":
            custom_value = st.text_input(f"Enter a custom value for {col}:",key=f"custom_{col}")
            handling_options[col] = custom_value
            
      
            
    
    #Apply transformations
    if st.button("Apply Changes"):
        df_filled = df.copy()
        
        for col,method in handling_options.items():
            if method == "Drop Column":
                df_filled.drop(columns=[col],inplace=True)
            elif method == "Fill with Mean":
                df_filled[col].fillna(df_filled[col].mean(),inplace=True)
                
            elif method == "Fill with Median":
                df_filled[col].fillna(df_filled[col].median(),inplace=True)
                
            elif method == "Fill with Mode":
                df_filled[col].fillna(df_filled[col].mode()[0],inplace=True)
                
    
        
        data_cleaned = True
        df = df_filled.copy()
        
   
        
    st.write("Dataframe Updated")
    st.write(df.sample())
    st.write(df.isnull().sum())
    
    #select column
    df_columns = list(df.columns)
    selected_column = st.selectbox("Set column as index: ",df_columns)
    
    #select plots
    plots = ["Histogram","Box Plot","Bar Chart","Pie Chart","KDE Chart"]
    selected_plot = st.selectbox("Set a plot: ",plots)
    if st.button("Plot it"):
        if selected_plot == "Histogram":
            try:
                fig = plot_histogram(df,selected_column)
                st.pyplot(fig)
            except Exception as e:
                st.write(f"{e} ")
        elif selected_plot == "Box Plot":
            try:
                fig = plot_box(df,selected_column)
                st.plotly_chart(fig)
            except Exception as e:
                st.write(f"{e} ")
        elif selected_plot == "Bar Chart":
            try:
                fig= plot_bar(df,selected_column)
                print(type(fig))
                st.plotly_chart(fig)
            except Exception as e:
                st.write(f"{e}")
                
        elif selected_plot == "Pie Chart":
            try:
                fig = plot_pie(df,selected_column)
                st.plotly_chart(fig)
            except Exception as e:
                st.write(e)
                
        elif selected_plot == "KDE Chart":
            try:
                fig = plot_kde_chart(df,selected_column)
                st.plotly_chart(fig)
            except Exception as e:
                st.write(e)
                
    st.title("Forecast Using SVM")
    st.write(df.describe())
    
    try:
        st.write(df.describe(include=['object','bool']))
    except Exception as e:
        st.write(e)

    feature_columns = st.multiselect("Choose Features",df_columns)
    
    if feature_columns:
        df.dropna(inplace=True)
        X = df[feature_columns]
        
        
        st.subheader("One hot encoding for categorical data")
        categorical_columns = st.multiselect("Choose columns",feature_columns)
        print(categorical_columns)
        
        
        if categorical_columns:
            X = one_hot_encode(X,categorical_columns)
            st.write(X.sample())
        
    
        label_column = st.selectbox("Choose a Label",df_columns)
        y = df[label_column]
        
        if st.button("Train with SVM"):
            
            model,accuracy,conf_matrix,class_report = train_and_evaluate_svm(X,y)
            st.subheader("Model Performance")
            st.write(f"**Accuracy:** {accuracy:.4f}")
            
            st.subheader("Confusion Matrix")
            st.write(pd.DataFrame(conf_matrix, index=model.classes_, columns=model.classes_))

            st.subheader("Classification Report")
            st.write(pd.DataFrame(class_report).transpose())
            
        if st.button("Train with Random Forest"):
            model,accuracy,conf_matrix,class_report = train_and_evaluate_rf(X,y)
            # Streamlit UI
            st.subheader("Model Performance")
            st.write(f"**Accuracy:** {accuracy:.4f}")
            
            st.subheader("Confusion Matrix")
            st.write(pd.DataFrame(conf_matrix, index=model.classes_, columns=model.classes_))

            st.subheader("Classification Report")
            st.write(pd.DataFrame(class_report).transpose())
        
        
       
        
        
        
        
            
        
        
    
    
