import streamlit as st
import pandas as pd
import streamlit.components.v1 as stc
from eda_app import run_eda_app
from ml_app import run_ml_app

def main():
    # Title of the app
    st.title("Classification Analysis with Streamlit")
    st.subheader("Perform EDA and Machine Learning on Classification Datasets")

    # Sidebar menu
    menu = ["Home", "EDA", "ML", "About"]
    choice = st.sidebar.selectbox("Navigation", menu)

    # Home section
    if choice == "Home":
        st.write("""
            ### Welcome to the Classification Analysis App
            This application allows you to perform exploratory data analysis (EDA) and machine learning on classification datasets.
            #### App Content
            - **EDA Section**: Perform Exploratory Data Analysis.
            - **ML Section**: Train and Evaluate Machine Learning Models.
        """)

        # Upload Button to Upload Dataset
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        # Check if a file is uploaded
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.session_state['df'] = df  # Store the dataframe in the session state
            st.write("Dataset loaded successfully!")


    # EDA section
    elif choice == "EDA":
        if 'df' in st.session_state:
            run_eda_app(st.session_state['df'])  # Pass the dataframe to the EDA app
        else:
            st.warning("Please upload a dataset in the Home section first.")

    # ML section
    elif choice == "ML":
        if 'df' in st.session_state:
            run_ml_app(st.session_state['df'])  # Pass the dataframe to the ML app
        else:
            st.warning("Please upload a dataset in the Home section first.")

    # About section
    else:
        st.write("### About")
        st.write("Submission by: Surbhi Kapoor")
        st.write("Langara ID: 100390953")
        st.write("This app is built using Streamlit for educational purposes.")

if __name__ == '__main__':
    main()