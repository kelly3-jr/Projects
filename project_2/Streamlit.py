import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load pre-trained model
model = joblib.load('stroke_prediction_model.pkl')

# Load dataset for visualizations
data = pd.read_csv('stroke_data.csv')

def main():
    # Set page configuration
    st.set_page_config(page_title="Brain Stroke Prediction", layout="wide")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose a mode:", ["Prediction", "Visualizations"])
    
    if app_mode == "Prediction":
        # Prediction page
        st.title("Brain Stroke Prediction App")
        st.write("Enter the details below to predict the likelihood of a stroke.")
        
        # Input fields
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        age = st.slider("Age", 0, 100, 30)
        hypertension = st.selectbox("Hypertension (High Blood Pressure)", [0, 1])
        heart_disease = st.selectbox("Heart Disease", [0, 1])
        ever_married = st.selectbox("Ever Married", ["No", "Yes"])
        work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
        residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
        avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, max_value=300.0, value=100.0)
        bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
        smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])
        
        # Map inputs to numeric values
        gender_map = {"Male": 1, "Female": 0, "Other": 2}
        ever_married_map = {"No": 0, "Yes": 1}
        work_type_map = {"Private": 2, "Self-employed": 3, "Govt_job": 1, "Children": 4, "Never_worked": 0}
        residence_type_map = {"Urban": 1, "Rural": 0}
        smoking_status_map = {"formerly smoked": 1, "never smoked": 2, "smokes": 3, "Unknown": 0}
        
        # Convert inputs
        inputs = np.array([
            gender_map[gender],
            age,
            hypertension,
            heart_disease,
            ever_married_map[ever_married],
            work_type_map[work_type],
            residence_type_map[residence_type],
            avg_glucose_level,
            bmi,
            smoking_status_map[smoking_status]
        ]).reshape(1, -1)
        
        if st.button("Predict"):
            prediction = model.predict(inputs)
            result = "The person is likely to have a stroke." if prediction[0] == 1 else "The person is unlikely to have a stroke."
            st.write(result)
            
            # Allow users to download prediction results
            results_df = pd.DataFrame({
                "Gender": [gender],
                "Age": [age],
                "Hypertension": [hypertension],
                "Heart Disease": [heart_disease],
                "Ever Married": [ever_married],
                "Work Type": [work_type],
                "Residence Type": [residence_type],
                "Avg Glucose Level": [avg_glucose_level],
                "BMI": [bmi],
                "Smoking Status": [smoking_status],
                "Prediction": ["Stroke" if prediction[0] == 1 else "No Stroke"]
            })
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Prediction Results",
                data=csv,
                file_name="prediction_results.csv",
                mime="text/csv",
            )
    
    elif app_mode == "Visualizations":
        # Visualizations page
        st.title("Data Visualizations")
        
        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        numeric_data = data.select_dtypes(include=["float64", "int64"])
        if not numeric_data.empty:
            corr_matrix = numeric_data.corr()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.write("No numeric data available for correlation heatmap.")
        
        # Feature importance
        st.subheader("Feature Importance")
        try:
            feature_importances = model.feature_importances_
            feature_names = ["Gender", "Age", "Hypertension", "Heart Disease", "Ever Married",
                             "Work Type", "Residence Type", "Avg Glucose Level", "BMI", "Smoking Status"]
            
            if len(feature_importances) == len(feature_names):
                importance_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": feature_importances
                }).sort_values(by="Importance", ascending=False)

                fig, ax = plt.subplots()
                sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis", ax=ax)
                ax.set_title("Feature Importance")
                st.pyplot(fig)
            else:
                st.write("Feature names and importances mismatch.")
        except AttributeError:
            st.write("Feature importance is not available for the loaded model.")

# Run the app
if __name__ == "__main__":
    main()

