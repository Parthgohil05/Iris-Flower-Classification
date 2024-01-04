import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

iris = load_iris()

# Separate out the features and target variable
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of the model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Display the model's performance
st.write(f"Accuracy: {model.score(X_test, y_test) * 100:.2f}%")

# Create a user interface to input the data
st.write("Enter the flower's measurements:")
sepal_length = st.number_input("Sepal Length (cm)", value=5.1)
sepal_width = st.number_input("Sepal Width (cm)", value=3.5)
petal_length = st.number_input("Petal Length (cm)", value=1.4)
petal_width = st.number_input("Petal Width (cm)", value=0.2)

# Predict the species of the flower
prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
st.write(f"The predicted species of the flower is: {iris['target_names'][prediction]}")