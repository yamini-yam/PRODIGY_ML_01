import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv("C:/Users/HP/Downloads/PRODIGY_DATASET/archive (2)/data.csv")

# Define features and target
X = df[['sqft_living', 'bedrooms', 'bathrooms']]
y = df['price']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'house_price_model.pkl')
