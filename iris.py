# import streamlit as st
# import numpy as np
# import pickle

# with open('dt_model.pkl', 'rb') as f:
#     model = pickle.load(f)

# st.title("Iris Species Prediction App")
# sepal_length = st.number_input("Sepal Length:", 0.0, 10.0, 5.0)
# sepal_width = st.number_input("Sepal Width:", 0.0, 10.0, 5.0)
# petal_length = st.number_input("Petal Length:", 0.0, 10.0, 5.0)
# petal_width = st.number_input("Petal Width:", 0.0, 10.0, 5.0)
# predict = st.button("Predict Features")


# if predict:
#     input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
#     prediction = model.predict(input_data)
#     species = ["Setosa", "Versicolor", "Virginica"]
#     st.success("The predicted Iris species is: ", species[prediction[0]])


import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open('dt_model.pkl', 'rb') as f:
    model = pickle.load(f)

# App title
st.title("Iris Species Prediction App")

# User inputs
sepal_length = st.number_input("Sepal Length:", 0.0, 10.0, 5.0)
sepal_width = st.number_input("Sepal Width:", 0.0, 10.0, 3.0)
petal_length = st.number_input("Petal Length:", 0.0, 10.0, 1.0)
petal_width = st.number_input("Petal Width:", 0.0, 10.0, 0.2)

# Predict button
predict = st.button("Predict Species")

if predict:
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)

    species = ["Setosa", "Versicolor", "Virginica"]

    # ✅ Correct usage of st.success
    st.success(f"The predicted Iris species is: {species[prediction[0]]}")

    
