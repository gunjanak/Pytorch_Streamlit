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
        critical_point = norm.ppf(st.session_state.alpha)  # Left-tailed critical point
    elif tail_type == "Right-Tailed":
        p_value = 1 - norm.cdf(z_score)
        critical_point = norm.ppf(1 - st.session_state.alpha)  # Right-tailed critical point
    else:  # Two-Tailed
        p_value = 2 * (1 - norm.cdf(abs(z_score)))
        critical_point = norm.ppf(1 - st.session_state.alpha / 2)  # Two-tailed critical point

    return z_score, p_value, critical_point

# Function to perform t-test
def t_test(pop_mean, sample_mean, sample_std, sample_size, tail_type):
    standard_error = sample_std / (sample_size ** 0.5)
    t_score = (sample_mean - pop_mean) / standard_error
    degrees_of_freedom = sample_size - 1

    if tail_type == "Left-Tailed":
        p_value = t.cdf(t_score, degrees_of_freedom)
        critical_point = t.ppf(st.session_state.alpha, degrees_of_freedom)  # Left-tailed critical point
    elif tail_type == "Right-Tailed":
        p_value = 1 - t.cdf(t_score, degrees_of_freedom)
        critical_point = t.ppf(1 - st.session_state.alpha, degrees_of_freedom)  # Right-tailed critical point
    else:  # Two-Tailed
        p_value = 2 * (1 - t.cdf(abs(t_score), degrees_of_freedom))
        critical_point = t.ppf(1 - st.session_state.alpha / 2, degrees_of_freedom)  # Two-tailed critical point

    return t_score, p_value, critical_point

# Function to plot the distribution and shade the critical region
def plot_distribution(test_type, tail_type, score, critical_point, alpha):
    fig, ax = plt.subplots()
    x = np.linspace(-4, 4, 1000)
    
    if test_type == "Z-Test":
        y = norm.pdf(x, 0, 1)
        label = "Standard Normal Distribution"
    else:
        degrees_of_freedom = st.session_state.sample_size - 1
        y = t.pdf(x, degrees_of_freedom)
        label = f"T-Distribution (df = {degrees_of_freedom})"

    ax.plot(x, y, label=label)
    ax.set_title(f"{test_type} - {tail_type}")
    ax.set_xlabel("Test Statistic")
    ax.set_ylabel("Probability Density")

    # Shade the critical region
    if tail_type == "Left-Tailed":
        x_fill = np.linspace(-4, critical_point, 1000)
        ax.fill_between(x_fill, norm.pdf(x_fill, 0, 1), color='red', alpha=0.5, label=f"Critical Region (α = {alpha})")
    elif tail_type == "Right-Tailed":
        x_fill = np.linspace(critical_point, 4, 1000)
        ax.fill_between(x_fill, norm.pdf(x_fill, 0, 1), color='red', alpha=0.5, label=f"Critical Region (α = {alpha})")
    else:  # Two-Tailed
        x_fill_left = np.linspace(-4, -critical_point, 1000)
        x_fill_right = np.linspace(critical_point, 4, 1000)
        ax.fill_between(x_fill_left, norm.pdf(x_fill_left, 0, 1), color='red', alpha=0.5, label=f"Critical Region (α = {alpha})")
        ax.fill_between(x_fill_right, norm.pdf(x_fill_right, 0, 1), color='red', alpha=0.5)

    # Add critical point and test statistic
    ax.axvline(critical_point, color='green', linestyle='--', label=f"Critical Point: {critical_point:.2f}")
    ax.axvline(score, color='blue', linestyle='--', label=f"Test Statistic: {score:.2f}")

    ax.legend()
    st.pyplot(fig)

# Streamlit app
st.title("Z-Test vs T-Test Calculator")

# User selects test type
test_type = st.selectbox("Select Test Type", ("Z-Test", "T-Test"))

# User selects tail type
tail_type = st.selectbox("Select Tail Type", ("Left-Tailed", "Right-Tailed", "Two-Tailed"))

# User inputs significance level
alpha = st.number_input("Significance Level (α)", value=0.05, min_value=0.01, max_value=0.10, step=0.01)
st.session_state.alpha = alpha

# Input fields based on test type
if test_type == "Z-Test":
    pop_mean = st.number_input("Population Mean", value=0.0)
    pop_std = st.number_input("Population Standard Deviation", value=1.0)
    sample_mean = st.number_input("Sample Mean", value=0.0)
    sample_size = st.number_input("Sample Size", value=30, min_value=1)
    st.session_state.sample_size = sample_size

    if st.button("Perform Z-Test"):
        z_score, p_value, critical_point = z_test(pop_mean, pop_std, sample_mean, sample_size, tail_type)
        st.write(f"Z-Score: {z_score:.4f}")
        st.write(f"P-Value: {p_value:.4f}")
        plot_distribution(test_type, tail_type, z_score, critical_point, alpha)

elif test_type == "T-Test":
    pop_mean = st.number_input("Population Mean", value=0.0)
    sample_mean = st.number_input("Sample Mean", value=0.0)
    sample_std = st.number_input("Sample Standard Deviation", value=1.0)
    sample_size = st.number_input("Sample Size", value=30, min_value=2)
    st.session_state.sample_size = sample_size

    if st.button("Perform T-Test"):
        t_score, p_value, critical_point = t_test(pop_mean, sample_mean, sample_std, sample_size, tail_type)
        st.write(f"T-Score: {t_score:.4f}")
        st.write(f"P-Value: {p_value:.4f}")
        plot_distribution(test_type, tail_type, t_score, critical_point, alpha)