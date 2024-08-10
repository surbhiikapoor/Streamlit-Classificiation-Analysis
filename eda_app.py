import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def run_eda_app(df):
    st.markdown(
        """
        <style>
        .subheader {
            font-size: 24px;
            color: #6a0dad;
            font-weight: bold;
            margin-bottom: 15px;
        }
        .button-container button {
            background-color: #6a0dad;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
            margin: 10px 0;
        }
        .button-container button:hover {
            background-color: #5a0c9c;
        }
        .checkbox-container .stCheckbox>label {
            color: #6a0dad;
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Custom styled subheaders
    st.markdown('<div class="subheader">Data Integrity Checks</div>', unsafe_allow_html=True)

    # Data Integrity Checks with buttons
    if st.button("Display Shape of the Dataset", key="shape"):
        st.write(f"**Shape of the dataset:** {df.shape}")

    if st.button("Display Missing Values in the Dataset", key="missing"):
        st.write("**Missing values in each column:**")
        st.write(df.isnull().sum())

    if st.button("Display First Few Rows of the Dataset", key="head"):
        st.write("**First few rows of the dataset:**")
        st.write(df.head())

    # Custom styled subheaders for EDA
    st.markdown('<div class="subheader">Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)

    # EDA - Histograms
    st.markdown('<div class="checkbox-container">', unsafe_allow_html=True)
    if st.checkbox("Show Histograms"):
        st.write("**Histograms for each feature:**")
        df.hist(figsize=(12, 8))
        st.pyplot(plt)
    st.markdown('</div>', unsafe_allow_html=True)

    # EDA - Pair Plots
    st.markdown('<div class="checkbox-container">', unsafe_allow_html=True)
    if st.checkbox("Show Pair Plot"):
        st.write("**Pair Plot:**")
        numeric_df = df.select_dtypes(include=['float64', 'int64'])  # Select only numeric columns
        if numeric_df.shape[1] > 1:  # Ensure there are at least 2 numeric columns
            sns.pairplot(numeric_df)
            st.pyplot(plt)
        else:
            st.warning("Pair plot requires at least two numeric columns.")
    st.markdown('</div>', unsafe_allow_html=True)

    # EDA - Correlation Heatmap
    st.markdown('<div class="checkbox-container">', unsafe_allow_html=True)
    if st.checkbox("Show Correlation Heatmap"):
        st.write("**Correlation Heatmap:**")
        numeric_df = df.select_dtypes(include=['float64', 'int64'])  # Select only numeric columns
        if numeric_df.shape[1] > 1:  # Ensure there are at least 2 numeric columns for correlation
            plt.figure(figsize=(10, 8))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
            st.pyplot(plt)
        else:
            st.warning("Correlation heatmap requires at least two numeric columns.")
    st.markdown('</div>', unsafe_allow_html=True)
