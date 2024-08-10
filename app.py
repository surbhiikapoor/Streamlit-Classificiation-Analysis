import streamlit as st
import pandas as pd
import streamlit.components.v1 as stc
from eda_app import run_eda_app
from ml_app import run_ml_app

# Custom CSS to enhance the appearance with a purple theme
st.markdown(
    """
    <style>
    .main {
        background-color: #f7f3f9;
    }
    .title {
        font-family: 'Arial';
        color: #6a0dad;
        font-size: 40px;
        text-align: center;
        margin-bottom: 20px;
    }
    .subtitle {
        font-family: 'Arial';
        color: #555555;
        font-size: 20px;
        text-align: center;
        margin-bottom: 30px;
    }
    .sidebar .sidebar-content {
        background-color: #6a0dad;
        color: white;
    }
    .stButton>button {
        background-color: #6a0dad;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #5a0c9c;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def main():
    # Main Title with custom style
    st.markdown('<div class="title">Classification Analysis App</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Perform EDA and Machine Learning on Classification Datasets</div>', unsafe_allow_html=True)

    # Sidebar menu
    st.sidebar.title("Navigation")
    menu = ["Home", "EDA", "ML", "About"]
    choice = st.sidebar.selectbox("", menu)

    # Home section
    if choice == "Home":
        st.write("""
            ### Welcome to the Classification Analysis App
            This application allows you to perform exploratory data analysis (EDA) and machine learning on classification datasets.
            #### App Content
            - **EDA Section**: Perform Exploratory Data Analysis.
            - **ML Section**: Train and Evaluate Machine Learning Models.
        """)

        # Columns layout for better alignment
        col1, col2 = st.columns(2)

        with col1:
            # Upload Button to Upload Dataset
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        with col2:
            st.write("")

        # Check if a file is uploaded
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.session_state['df'] = df  # Store the dataframe in the session state
            st.success("Dataset loaded successfully!")


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
        st.write("#### Submission by: Surbhi Kapoor")
        st.write("#### Langara ID: 100390953")
        st.write("This app is built using Streamlit for educational purposes.")

if __name__ == '__main__':
    main()
