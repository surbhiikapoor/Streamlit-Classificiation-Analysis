import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def run_eda_app(df):
    st.subheader("Exploratory Data Analysis (EDA)")

    # Data Integrity Checks
    st.subheader("Data Integrity Checks")

    if st.button("Display Shape of the Dataset"):
        st.write(f"Shape of the dataset: {df.shape}")

    if st.button("Display Missing Values in the Dataset"):
        st.write("Missing values in each column:")
        st.write(df.isnull().sum())

    if st.button("Display First Few Rows of the Dataset"):
        st.write("First few rows of the dataset:")
        st.write(df.head())

    # EDA - Histograms
    st.subheader("Histograms")
    if st.checkbox("Show Histograms"):
        st.write("Histograms for each feature:")
        df.hist(figsize=(12, 8))
        st.pyplot(plt)

    # EDA - Pair Plots
    st.subheader("Pair Plots")
    if st.checkbox("Show Pair Plot"):
        st.write("Pair Plot:")
        numeric_df = df.select_dtypes(include=['float64', 'int64'])  # Select only numeric columns
        if numeric_df.shape[1] > 1:  # Ensure there are at least 2 numeric columns
            sns.pairplot(numeric_df)
            st.pyplot(plt)
        else:
            st.warning("Pair plot requires at least two numeric columns.")

    # EDA - Correlation Heatmap
    st.subheader("Correlation Heatmap")
    if st.checkbox("Show Correlation Heatmap"):
        st.write("Correlation Heatmap:")
        numeric_df = df.select_dtypes(include=['float64', 'int64'])  # Select only numeric columns
        if numeric_df.shape[1] > 1:  # Ensure there are at least 2 numeric columns for correlation
            plt.figure(figsize=(10, 8))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
            st.pyplot(plt)
        else:
            st.warning("Correlation heatmap requires at least two numeric columns.")