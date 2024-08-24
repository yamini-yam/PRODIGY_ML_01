import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('house_price_model.pkl')

def predict_price(sqft_living, bedrooms, bathrooms):
    """Predict house price based on input features."""
    input_features = pd.DataFrame([[sqft_living, bedrooms, bathrooms]], columns=['sqft_living', 'bedrooms', 'bathrooms'])
    predicted_price = model.predict(input_features)[0]
    return predicted_price

def main():
    # Streamlit UI
    st.title('House Price Prediction')

    st.write("Enter the details of the house to get the estimated price:")

    # Sidebar for user input
    sqft_living = st.sidebar.number_input('Square Footage', min_value=0, value=1500)
    bedrooms = st.sidebar.number_input('Number of Bedrooms', min_value=0, value=3)
    bathrooms = st.sidebar.number_input('Number of Bathrooms', min_value=0, value=2)

    st.write(f"Inputs - Square Footage: {sqft_living}, Bedrooms: {bedrooms}, Bathrooms: {bathrooms}")

    # Button to predict
    if st.button('Predict Price'):
        # Predict price based on user input
        predicted_price = predict_price(sqft_living, bedrooms, bathrooms)
        
        # Display the result
        st.write(f"**Predicted Price:** ${predicted_price:,.2f}")

if __name__ == "__main__":
    main()
