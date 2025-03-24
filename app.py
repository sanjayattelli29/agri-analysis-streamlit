import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def get_predictions(features):
    API_URL = "https://agri-assist-models.onrender.com/predict"
    response = requests.post(API_URL, json={"features": features})
    
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"API Error: {response.status_code} - {response.text}"}

# Streamlit UI
st.title("Agri-Assist Model Predictor 🌾")
st.write("Enter the values for the following features to get predictions.")

# Input fields (empty by default, users must enter values)
N = st.number_input("Nitrogen (N)", min_value=0.0, step=0.1, placeholder="Enter Nitrogen value")
P = st.number_input("Phosphorus (P)", min_value=0.0, step=0.1, placeholder="Enter Phosphorus value")
K = st.number_input("Potassium (K)", min_value=0.0, step=0.1, placeholder="Enter Potassium value")
temp = st.number_input("Temperature (°C)", min_value=-50.0, max_value=100.0, step=0.1, placeholder="Enter Temperature")
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1, placeholder="Enter Humidity")
pH = st.number_input("pH Level", min_value=0.0, max_value=14.0, step=0.1, placeholder="Enter pH Level")
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, step=0.1, placeholder="Enter Rainfall")

# Button to submit data
if st.button("Predict Crop"):
    features = [N, P, K, temp, humidity, pH, rainfall]
    result = get_predictions(features)
    
    if "error" in result:
        st.error(result["error"])
    else:
        st.subheader("Model Predictions")
        for model, prediction in result["predictions"].items():
            st.write(f"**{model}:** {prediction}")
        
        st.subheader("Model Performance Metrics")
        df_metrics = pd.DataFrame(result["metrics"]).T
        st.dataframe(df_metrics)
        
        # Graphical Analysis
        st.subheader("Graphical Analysis")
        st.write("Comparing model performance using different metrics.")
        
        metrics_to_visualize = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Log Loss', 'MAE', 'MSE', 'RMSE', 'R² Score']
        for metric in metrics_to_visualize:
            fig, ax = plt.subplots(figsize=(7, 3))  # Reduced figure size
            sns.barplot(x=df_metrics.index, y=df_metrics[metric], ax=ax, hue=df_metrics.index, palette="coolwarm", legend=False)
            ax.set_title(f"{metric} Comparison Across Models")
            ax.set_xlabel("Models")
            ax.set_ylabel(metric)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        # Line graph for overall performance trends
        st.subheader("Overall Performance Trends")
        fig, ax = plt.subplots(figsize=(8, 4))
        df_metrics[['Accuracy', 'Precision', 'Recall', 'F1-Score']].plot(kind='line', marker='o', ax=ax)
        plt.title("Model Performance Trends")
        plt.xlabel("Models")
        plt.ylabel("Score")
        plt.xticks(range(len(df_metrics)), df_metrics.index, rotation=45)
        plt.legend()
        st.pyplot(fig)
        
        # Pie Chart based on Accuracy
        st.subheader("Model Accuracy Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(df_metrics['Accuracy'], labels=df_metrics.index, autopct='%1.1f%%', colors=sns.color_palette("coolwarm", len(df_metrics)))
        ax.set_title("Accuracy Distribution Among Models")
        st.pyplot(fig)
        
        st.subheader("Final Recommendation")
        st.success(result["final_recommendation"])
        
        st.write("**Why this model is best?**")
        best_model = df_metrics['Accuracy'].idxmax()
        st.write(f"The best performing model is **{best_model}**, which has the highest accuracy. It effectively balances precision, recall, and F1-score, making it the most reliable for this classification problem.")

st.write("Developed by Sanjay 🚀")
