import streamlit as st
import pickle
import os
import numpy as np

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the model file
model_path = os.path.join(current_dir, "lr_model.pkl")

# Load pre-trained model and vectorizer
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Streamlit UI
def main():
    st.title("Phân Loại Loài Hoa Iris")

    # Display an image
    image_path = os.path.join(current_dir, "iris.png")
    st.image(image_path, use_column_width=True)

    # Input parameters
    sepal_length = st.slider("Chọn kích thước sepal length:", 4.0, 8.0, 5.0)
    sepal_width = st.slider("Chọn kích thước sepal width:", 2.0, 4.5, 3.0)

    # Make prediction
    features = np.array([[sepal_length, sepal_width]])
    prediction = model.predict(features)[0]
    species_mapping = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    predicted_species = species_mapping[prediction]

    # Display result
    st.write(f"Kích thước sepal length: {sepal_length}")
    st.write(f"Kích thước sepal width: {sepal_width}")
    st.write(f"Loài hoa dự đoán: {predicted_species}")

if __name__ == "__main__":
    main()
