import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# Define the model and scaler loading code inside Streamlit app
def run_app():
    # Load the trained model (ensure it's in the same directory as this script)
    with open('svm_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Load the scaler (make sure the scaler is saved as 'scaler.pkl' and located in the same directory)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Define the features that the model expects
    features = [
        'current_month_debit', 'previous_month_debit', 'current_balance', 
        'previous_month_credit', 'current_month_credit', 'average_monthly_balance_prevQ', 
        'average_monthly_balance_prevQ2', 'previous_month_balance', 'current_month_balance', 
        'branch_code', 'dependents', 'age'
    ]
    
    # Order of the input data for text input
    feature_order = ', '.join(features)

    # Title of the web app
    st.title('SVM Model Prediction')

    # Input field for the user to enter comma-separated values
    user_input = st.text_input(f'Enter values for features ({feature_order})', '')

    # Create a button for prediction
    if st.button('Predict'):
        if user_input:
            # Split the input data by commas and convert it to a list of floats
            try:
                input_values = [float(x.strip()) for x in user_input.split(',')]
            except ValueError:
                st.error("Please enter valid numeric values.")
                return

            if len(input_values) != len(features):
                st.error(f"Incorrect number of values entered. You must enter {len(features)} values.")
            else:
                # Create a DataFrame from the inputs
                input_data = pd.DataFrame([input_values], columns=features)

                # Check if the scaler is fitted
                try:
                    # Scale the input data (as done during training)
                    input_data_scaled = scaler.transform(input_data)  # Use transform here, not fit_transform

                    # Predict using the loaded model
                    prediction = model.predict(input_data_scaled)

                    # Display the prediction
                    if prediction == 1:
                        st.write("Prediction: Churn (1)")
                    else:
                        st.write("Prediction: No Churn (0)")

                except Exception as e:
                    st.error(f"Error during scaling or prediction: {str(e)}")
        else:
            st.error("Please enter the feature values.")

# Set the app to run
if __name__ == "__main__":
    run_app()
