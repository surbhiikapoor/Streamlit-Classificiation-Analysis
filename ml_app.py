import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np


def run_ml_app(df):
    st.subheader("Model Selection and Training")

    # Selecting Features and Target
    features = st.multiselect("Select features for the model", df.columns.tolist(), default=df.columns.tolist()[:-1])
    target = st.selectbox("Select target variable", df.columns.tolist(), index=len(df.columns.tolist()) - 1)

    X = df[features]
    y = df[target]

    # Check if target is continuous and convert to categorical if necessary
    if y.dtype in ['float64', 'int64']:
        st.warning("Target variable is continuous. Converting to categorical for classification.")
        bins = st.slider("Select number of bins", 2, 10, 4)
        y = pd.cut(y, bins=bins, labels=False)  # Discretize the target variable

    # Data Scaling Options
    scaling_option = st.selectbox("Choose Scaling Method", ["None", "StandardScaler", "MinMaxScaler"])
    if scaling_option == "StandardScaler":
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    elif scaling_option == "MinMaxScaler":
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

    # Split the dataset into training and testing sets
    if st.button("Split Data"):
        test_size = st.slider("Test Size (percentage)", 10, 50, 30)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=42)
        st.session_state['X_train'] = X_train
        st.session_state['X_test'] = X_test
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test
        st.success(f"Data split into training and testing sets with {test_size}% for testing.")

    # Model Selection and Hyperparameter Tuning
    model_type = st.selectbox("Choose a model", ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors"])

    if model_type == "Logistic Regression":
        model = LogisticRegression()
    elif model_type == "Decision Tree":
        max_depth = st.slider("Max Depth of Decision Tree", 1, 20, value=5)
        model = DecisionTreeClassifier(max_depth=max_depth)
    elif model_type == "K-Nearest Neighbors":
        k = st.slider("Number of Neighbors (k)", 1, 20, value=5)
        model = KNeighborsClassifier(n_neighbors=k)

    # Training the Model
    if st.button("Train Model"):
        if 'X_train' in st.session_state and 'y_train' in st.session_state:
            model.fit(st.session_state['X_train'], st.session_state['y_train'])
            st.session_state['model'] = model
            st.success("Model trained successfully!")
        else:
            st.warning("Please split the data first.")

    # Prediction
    if st.button("Predict"):
        if 'model' in st.session_state:
            y_pred = st.session_state['model'].predict(st.session_state['X_test'])
            st.session_state['y_pred'] = y_pred
            st.success("Prediction done!")
        else:
            st.warning("Please train the model first.")

    # Evaluation Metrics
    st.subheader("Evaluate Model Performance")

    if 'y_pred' in st.session_state:
        if st.button("Show Accuracy"):
            accuracy = accuracy_score(st.session_state['y_test'], st.session_state['y_pred'])
            st.write(f"Model Accuracy: {accuracy:.2f}")

        if st.button("Show Precision"):
            precision = precision_score(st.session_state['y_test'], st.session_state['y_pred'], average='weighted')
            st.write(f"Model Precision: {precision:.2f}")

        if st.button("Show Recall"):
            recall = recall_score(st.session_state['y_test'], st.session_state['y_pred'], average='weighted')
            st.write(f"Model Recall: {recall:.2f}")

        if st.button("Show Confusion Matrix"):
            st.write("Confusion Matrix:")
            st.write(confusion_matrix(st.session_state['y_test'], st.session_state['y_pred']))

        if st.button("Show ROC Curve"):
            y_test = st.session_state['y_test']
            if len(np.unique(y_test)) == 2:  # Binary classification
                y_pred_prob = st.session_state['model'].predict_proba(st.session_state['X_test'])[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label="ROC curve (area = {:.2f})".format(auc(fpr, tpr)))
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.legend(loc="lower right")
                st.pyplot(plt)
            else:  # Multi-class classification
                y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
                n_classes = y_test_bin.shape[1]
                y_score = st.session_state['model'].predict_proba(st.session_state['X_test'])

                # Compute ROC curve and ROC area for each class
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])

                # Plot all ROC curves
                plt.figure(figsize=(8, 6))
                colors = plt.cm.get_cmap('Set1', n_classes)
                for i, color in zip(range(n_classes), colors.colors):
                    plt.plot(fpr[i], tpr[i], color=color, lw=2,
                             label='ROC curve of class {0} (area = {1:0.2f})'
                                   ''.format(i, roc_auc[i]))

                plt.plot([0, 1], [0, 1], 'k--', lw=2)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.legend(loc="lower right")
                st.pyplot(plt)
    else:
        st.warning("Please perform predictions first.")

    # Download the Predictions as a CSV file
    if st.button("Download Predictions as CSV"):
        if 'y_pred' in st.session_state:
            results_df = pd.DataFrame(st.session_state['X_test'], columns=features)
            results_df['Actual'] = st.session_state['y_test'].reset_index(drop=True)
            results_df['Predicted'] = st.session_state['y_pred']
            csv = results_df.to_csv(index=False)
            st.download_button("Download CSV", csv, "predictions.csv", "text/csv")
        else:
            st.warning("Please perform predictions first.")