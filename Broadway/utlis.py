import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def load_data():
    st.write("Upload a csv file")
    uploaded_file = st.file_uploader("Choose a file",'csv')
    use_example_file = st.checkbox("Use example file",False,help="Use in-built example file for demo")

    status = False
    if use_example_file:
        uploaded_file = "default_file.csv"
        status = True
    
    if uploaded_file:
        #st.write(uploaded_file)
        if(uploaded_file == None):
            status = False
        else:
            status = True
    to_return = [uploaded_file,status]

    return to_return

def read_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df


def plot_histogram(df, column,bins=5,figsize=(4,2)):
    """Plots a histogram for a given column in a DataFrame."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(df[column], bins=5, edgecolor='black', alpha=0.7)
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    ax.set_title(f"{column} Distribution")
    return fig

def plot_box(df,column,width=800,height=600):
    fig = px.box(df,y=column,title=f"Box plot of {column}")
    fig.update_layout(yaxis_title=column,width=width, 
                      height=height,template="plotly_dark")
    print(type(fig))
    return fig

def plot_bar(df,column,width=800,height=600):
    value_counts = df[column].value_counts().reset_index()
    value_counts.columns = [column,"Count"]
    fig = px.bar(value_counts,
                 x=column,y="Count",
                 title=f"Bar chart of {column}",
                 labels={column:column,"Count":"frequency"},
                 text = "Count")
    fig.update_layout(width=width,height=height,template="plotly_dark",xaxis_title=column,
                      yaxis_title="Count")
    return fig

def plot_pie(df,column,width=800,height=600):
    value_counts = df[column].value_counts().reset_index()
    print(value_counts)
    value_counts.columns = [column,"Count"]
    print(value_counts)
    fig = px.pie(value_counts,names=column,values="Count",title=f"Distribution of {column}",
                 hole=0.3)
    return fig

def plot_kde_chart(df,column,width=600,height=400):
    fig = ff.create_distplot([df[column].dropna()],[column],
                             show_hist=False,show_rug=True)
    fig.update_layout(title=f"KDE Plot of {column}",
                      width=width,
                      height=height,
                      template="plotly_dark",
                      xaxis_title=column,
                      yaxis_title="Density")
    return fig
    
    
    
    
    
    
def one_hot_encode(df,columns):
    return pd.get_dummies(df,columns=columns,drop_first=True)




def train_and_evaluate_svm(X, y, test_size=0.4, random_state=42):
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Train SVM model
    model = SVC(kernel="rbf", C=1.0, random_state=random_state)  # You can change kernel and hyperparameters
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test,y_pred, output_dict=True)  # Convert to dict for better Streamlit display

    

    return model,accuracy,conf_matrix,class_report




def train_and_evaluate_rf(X, y, test_size=0.2, random_state=42, n_estimators=100):
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test,y_pred, output_dict=True)  # Convert to dict for Streamlit display

    

    return model,accuracy,conf_matrix,class_report