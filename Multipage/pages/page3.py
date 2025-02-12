import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff


st.title("Hypothesis Testing")
type_of_test = ['Z-test','T-test']
selected_test = st.selectbox("Select Hypothesis Test:",type_of_test)
print(selected_test)