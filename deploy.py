# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib

# Function to load and preprocess data
def load_and_preprocess_data():
    try:
        data = pd.read_csv('smoke_detection_iot.csv')

        # Identify and encode categorical columns
        catcol = [col for col in data.columns if data[col].dtype == "object"]
        le = LabelEncoder()
        for col in catcol:
            data[col] = le.fit_transform(data[col])

        # Rename columns for simplicity
        data.rename(columns={
            "Temperature[C]": "Temperature",
            "Humidity[%]": "Humidity",
            "TVOC[ppb]": "TVOC",
            "eCO2[ppm]": "eCO2",
            "Pressure[hPa]": "Pressure"
        }, inplace=True)

        # Select relevant columns
        data = data[["Temperature", "Humidity", "TVOC", "eCO2", "Raw H2", "Raw Ethanol", "Pressure", "PM1.0", "PM2.5", "NC0.5", "NC1.0", "NC2.5", "Fire Alarm"]]

        # Extract features (x) and target variable (y)
        x = data.drop(["Fire Alarm"], axis=1)
        y = data["Fire Alarm"]

        # Standardize the features
        sc = StandardScaler()
        x = sc.fit_transform(x)

        # Split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        return x_train, x_test, y_train, y_test, sc

    except Exception as e:
        st.error(f"An error occurred while loading and preprocessing data: {e}")
        return None, None, None, None, None

# Function to train the XGBoost classifier
def train_xgboost_classifier(x_train, y_train):
    try:
        xgb = XGBClassifier(use_label_encoder=False)
        xgb.fit(x_train, y_train)
        return xgb
    except Exception as e:
        st.error(f"An error occurred during XGBoost classifier training: {e}")
        return None

# Function to evaluate the XGBoost classifier
def evaluate_classifier(model, x_test, y_test):
    try:
        predictions = model.predict(x_test)
        confusion = confusion_matrix(y_test, predictions)
        report = classification_report(y_test, predictions)
        accuracy = round(accuracy_score(y_test, predictions) * 100, 2)
        return confusion, report, accuracy
    except Exception as e:
        st.error(f"An error occurred during classifier evaluation: {e}")
        return None, None, None

# Function to save the trained model
def save_model(model, filename='xgboost_model.joblib'):
    try:
        joblib.dump(model, filename)
    except Exception as e:
        st.error(f"An error occurred while saving the model: {e}")

# Function to load the saved model
def load_saved_model(filename='xgboost_model.joblib'):
    try:
        return joblib.load(filename)
    except Exception as e:
        st.error(f"An error occurred while loading the saved model: {e}")
        return None

# Function to make predictions using the loaded model
def make_predictions(model, new_data, scaler):
    try:
        new_features = scaler.transform(new_data)
        return model.predict(new_features)
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None

# Function to display bar plots for input features
def display_feature_plots(new_data):
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        new_data.T.plot(kind='bar', legend=False, ax=ax)
        plt.title("Input Features")
        plt.xlabel("Features")
        plt.ylabel("Values")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"An error occurred during feature plots display: {e}")

# Function to create the Streamlit App
def create_streamlit_app():
    try:
        st.sidebar.header("Enter New Data:")

        # User input
        temperature = st.sidebar.number_input("Temperature [C]:", value=20)
        humidity = st.sidebar.number_input("Humidity [%]:", value=57.6)
        tvoc = st.sidebar.number_input("TVOC [ppb]:", value=0)
        eco2 = st.sidebar.number_input("eCO2 [ppm]:", value=400)
        raw_h2 = st.sidebar.number_input("Raw H2:", value=12306)
        raw_ethanol = st.sidebar.number_input("Raw Ethanol:", value=18520)
        pressure = st.sidebar.number_input("Pressure [hPa]:", value=939.735)
        pm1 = st.sidebar.number_input("PM1.0:", value=0)
        pm25 = st.sidebar.number_input("PM2.5:", value=0)
        nc05 = st.sidebar.number_input("NC0.5:", value=0)
        nc1 = st.sidebar.number_input("NC1.0:", value=0)
        nc25 = st.sidebar.number_input("NC2.5:", value=0)

        # Create a DataFrame with user input
        new_data = pd.DataFrame({
            'Temperature[C]': [temperature],
            'Humidity[%]': [humidity],
            'TVOC[ppb]': [tvoc],
            'eCO2[ppm]': [eco2],
            'Raw H2': [raw_h2],
            'Raw Ethanol': [raw_ethanol],
            'Pressure[hPa]': [pressure],
            'PM1.0': [pm1],
            'PM2.5': [pm25],
            'NC0.5': [nc05],
            'NC1.0': [nc1],
            'NC2.5': [nc25],
        })

        # Load the saved model and scaler
        loaded_model = load_saved_model()
        scaler = load_and_preprocess_data()[-1]

        # Make predictions using the loaded model
        prediction = make_predictions(loaded_model, new_data, scaler)

        # Display the prediction result
        st.subheader("Prediction:")
        if prediction is not None and prediction[0] == 1:
            st.success("Fire Alarm: Yes")
        else:
            st.info("Fire Alarm: No")

        # Display bar plots for the input features
        st.subheader("Input Features:")
        display_feature_plots(new_data)

    except Exception as e:
        st.error(f"An error occurred while creating the Streamlit app: {e}")

# Streamlit App main function
def main():
    try:
        st.sidebar.title("Fire Alarm System")
        st.sidebar.subheader("This is a demo of the Fire Alarm System, predicting the probability of fire based on real-time sensor data.")
        st.sidebar.write("The system uses Python, Streamlit, Tensorflow, and Keras.")
        st.sidebar.write("The neural network model has 3 hidden layers and 1 output layer.")

        st.title("🔥 Fire Alarm System 🔥")

        # Load and preprocess data
        x_train, x_test, y_train, y_test, scaler = load_and_preprocess_data()

        # Train the XGBoost classifier
        xgboost_model = train_xgboost_classifier(x_train, y_train)

        # Evaluate the XGBoost classifier
        confusion_matrix, classification_report, accuracy = evaluate_classifier(xgboost_model, x_test, y_test)

        # Save the trained XGBoost model
        save_model(xgboost_model)

        st.info(f"Accuracy: {accuracy}%")

        # Create and display the Streamlit App
        create_streamlit_app()

    except Exception as e:
        st.error(f"An error occurred in the main function: {e}")

if __name__ == "__main__":
    main()
