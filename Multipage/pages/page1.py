#imports
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff

import statsmodels.api as sm
from sklearn.model_selection import train_test_split


# Function to detect outliers using IQR method
def detect_outliers(data, column):
    Q1 = np.percentile(data[column], 25)
    Q3 = np.percentile(data[column], 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (data[column] < lower_bound) | (data[column] > upper_bound)


#Forward selection function
def forward_selection(X,y,significance_level=0.05):
    initial_features = list(X.columns)
    selected_features = []
    
    while len(initial_features)>0:
        best_pval = significance_level
        best_feature = None
        
        for feature in initial_features:
            model = sm.OLS(y,sm.add_constant(pd.DataFrame(X[selected_features+[feature]]))).fit()
            pval = model.pvalues[feature]
            
            if pval < best_pval:
                best_pval = pval
                best_feature = feature
                
        if best_feature is not None:
            selected_features.append(best_feature)
            initial_features.remove(best_feature)
            
        else:
            break
        
    return selected_features
            

st.title("Bank data analysis")
#Read the data
df = pd.read_csv("financial_data.csv")

#set date as index
df.set_index("Date",inplace=True)

#Display the data
st.dataframe(df)

#Plot the data
selected_column = st.selectbox("Select a column to plot:",df.columns)
st.line_chart(df[selected_column])

#Explore the data
st.subheader("Description of the data")
st.write(df.describe())

st.subheader("Null")
st.write(df.isnull().sum())


#Fill missing values with mean value
df = df.fillna(df.median(numeric_only=True))
st.subheader("Fill missing values with mean value")
st.write(df)

#Find the outliers

st.subheader("Outlier Detection & Plotting")

# Dropdown to select a column
selected_column = st.selectbox("Select a column to analyze:", df.columns)
# Detect outliers
df_outlier = df.copy()
df_outlier["Outlier"] = detect_outliers(df_outlier, selected_column)
# Plot data using Plotly
fig = px.scatter(df_outlier, x=df_outlier.index, y=selected_column, 
                 color=df_outlier["Outlier"].map({False: "Normal", True: "Outlier"}),
                 title=f"Outlier Detection in {selected_column}",
                 labels={"color": "Data Type"},
                 color_discrete_map={"Normal": "blue", "Outlier": "red"})

# Show plot
st.plotly_chart(fig)

# Show detected outliers
st.write("### Outliers Detected:")
st.write(df_outlier[df_outlier["Outlier"]])

#Find the correlation
st.subheader("Correlation")
st.write(df.corr())

# Streamlit App
st.title("Correlation Matrix Heatmap")


# Compute correlation matrix
corr_matrix = df.corr()

# Create correlation heatmap using Plotly
fig = ff.create_annotated_heatmap(
    z=corr_matrix.values,
    x=list(corr_matrix.columns),
    y=list(corr_matrix.index),
    colorscale="Viridis",
    annotation_text=np.round(corr_matrix.values, 2),
    showscale=True
)

# Display plot
st.plotly_chart(fig)

# Show correlation matrix values
st.write("### Correlation Matrix Values")
st.dataframe(corr_matrix)




#Forward selection techniques
df_cleaned = df.dropna()
X = df_cleaned.drop(columns=["ROE"])
y = df_cleaned["ROE"]

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)
selected_features = forward_selection(X_train,y_train)

st.write(selected_features)
final_model = sm.OLS(y_train,sm.add_constant(X_train[selected_features])).fit()
st.write(final_model.summary())

predictions = final_model.predict(sm.add_constant(X_test[selected_features]))


output = pd.DataFrame({
    "Predictions": predictions,
    "ROE": y_test
})

st.write(output)

