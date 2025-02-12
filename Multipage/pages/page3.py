import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
from scipy.stats import norm,t

# Function to perform z-test
def z_test(pop_mean, pop_std, sample_mean, sample_size, tail_type):
    standard_error = pop_std / (sample_size ** 0.5)
    z_score = (sample_mean - pop_mean) / standard_error

    if tail_type == "Left-Tailed":
        p_value = norm.cdf(z_score)
    elif tail_type == "Right-Tailed":
        p_value = 1 - norm.cdf(z_score)
    else:  # Two-Tailed
        p_value = 2 * (1 - norm.cdf(abs(z_score)))

    return z_score, p_value

# Function to perform t-test
def t_test(pop_mean, sample_mean, sample_std, sample_size, tail_type):
    standard_error = sample_std / (sample_size ** 0.5)
    t_score = (sample_mean - pop_mean) / standard_error
    degrees_of_freedom = sample_size - 1

    if tail_type == "Left-Tailed":
        p_value = t.cdf(t_score, degrees_of_freedom)
    elif tail_type == "Right-Tailed":
        p_value = 1 - t.cdf(t_score, degrees_of_freedom)
    else:  # Two-Tailed
        p_value = 2 * (1 - t.cdf(abs(t_score), degrees_of_freedom))

    return t_score, p_value

st.title("Hypothesis Testing")
type_of_test = ['Z-test','T-test']
selected_test = st.selectbox("Select Hypothesis Test:",type_of_test)
# User selects tail type
tail_type = st.selectbox("Select Tail Type", ("Left-Tailed", "Right-Tailed", "Two-Tailed"))


# Input fields based on test type
if selected_test == "Z-test":
    pop_mean = st.number_input("Population Mean", value=0.0)
    pop_std = st.number_input("Population Standard Deviation", value=1.0)
    sample_mean = st.number_input("Sample Mean", value=0.0)
    sample_size = st.number_input("Sample Size", value=30, min_value=1)

    if st.button("Perform Z-Test"):
        z_score, p_value = z_test(pop_mean, pop_std, sample_mean, sample_size, tail_type)
        st.write(f"Z-Score: {z_score:.4f}")
        st.write(f"P-Value: {p_value:.4f}")

elif selected_test == "T-test":
    pop_mean = st.number_input("Population Mean", value=0.0)
    sample_mean = st.number_input("Sample Mean", value=0.0)
    sample_std = st.number_input("Sample Standard Deviation", value=1.0)
    sample_size = st.number_input("Sample Size", value=30, min_value=2)

    if st.button("Perform T-Test"):
        t_score, p_value = t_test(pop_mean, sample_mean, sample_std, sample_size, tail_type)
        st.write(f"T-Score: {t_score:.4f}")
        st.write(f"P-Value: {p_value:.4f}")