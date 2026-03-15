import streamlit as st

# ------------------------------------------------

# PAGE TITLE

# ------------------------------------------------

st.title("About")

st.write(
"""
This **Analytics Dashboard** is an interactive web application built using **Streamlit**.
It demonstrates how **data analysis, visualization, and machine learning models**
can be deployed into a clean and user-friendly web interface.
"""
)

# ------------------------------------------------

# FEATURES

# ------------------------------------------------

st.subheader("Dashboard Features")

st.markdown(
"""
• Interactive analytics dashboard
• Data exploration and filtering tools
• Built-in visualizations and charts
• Iris species machine learning prediction
• Dataset download capability
• Modern responsive user interface
"""
)

# ------------------------------------------------

# TECHNOLOGY STACK

# ------------------------------------------------

st.subheader("Technology Stack")

col1, col2 = st.columns(2)

with col1:
st.markdown(
"""
**Framework**

* Streamlit

**Programming Language**

* Python
  """
  )

with col2:
st.markdown(
"""
**Libraries**

* Pandas
* NumPy
* Scikit-learn
  """
  )

# ------------------------------------------------

# MACHINE LEARNING MODEL

# ------------------------------------------------

st.subheader("Machine Learning Model")

st.markdown(
"""
This dashboard includes an **Iris Species Prediction Model** trained on the
classic **Iris dataset**, one of the most well-known datasets in machine learning.

The model predicts the species of a flower using four measurements:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

Possible predictions:

* Setosa
* Versicolor
* Virginica
  """
  )

# ------------------------------------------------

# PROJECT STRUCTURE

# ------------------------------------------------

st.subheader("Project Structure")

st.code(
"""
StreamLit
│
├── app.py
├── assets/
│   └── style.css
│
├── models/
│   └── dt_model.pkl
│
├── pages/
│   ├── 1_Data_Explorer.py
│   ├── 2_Iris_Predictor.py
│   └── 3_About.py
│
├── requirements.txt
└── README.md
"""
)

# ------------------------------------------------

# AUTHOR

# ------------------------------------------------

st.subheader("Author")

st.markdown(
"""
**Muhammed Faris T M**

Data Science • Analytics • Machine Learning

This project was created as part of a **data science portfolio**
to demonstrate skills in:

• Data analytics
• Machine learning deployment
• Streamlit dashboard development
"""
)

# ------------------------------------------------

# CONTACT

# ------------------------------------------------

st.subheader("Connect")

st.markdown(
"""
**LinkedIn**
http://www.linkedin.com/in/muhammed-faris-tm-ab1233196
"""
)

# ------------------------------------------------

# FOOTER

# ------------------------------------------------

st.write("---")

st.caption("Built with Streamlit • Analytics Dashboard")
