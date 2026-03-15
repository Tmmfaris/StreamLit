import streamlit as st


# ------------------------------------------------
# PAGE TITLE
# ------------------------------------------------
st.title("About This Project")


st.write(
    """
This **Data Analytics Dashboard** is an interactive web application built using **Streamlit**.  
It demonstrates how data science models and analytics tools can be deployed into a
user-friendly web interface.
"""
)


# ------------------------------------------------
# PROJECT FEATURES
# ------------------------------------------------
st.subheader("Features")

st.markdown(
    """
- Interactive analytics dashboard
- Data exploration and filtering
- Built-in data visualizations
- Machine Learning prediction (Iris species)
- Downloadable datasets
- Modern responsive UI
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
**Frontend**
- Streamlit
- Custom CSS Styling
"""
    )

with col2:
    st.markdown(
        """
**Backend / Data**
- Python
- Pandas
- NumPy
- Scikit-learn
"""
    )


# ------------------------------------------------
# MACHINE LEARNING MODEL
# ------------------------------------------------
st.subheader("Machine Learning Model")

st.markdown(
    """
The dashboard includes an **Iris Species Prediction Model** trained on the
classic Iris dataset.

**Input Features**
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

**Output**
- Predicted Iris Species
  - Setosa
  - Versicolor
  - Virginica

The model demonstrates how machine learning can be integrated into a
web-based analytics application.
"""
)


# ------------------------------------------------
# PROJECT STRUCTURE
# ------------------------------------------------
st.subheader("Project Structure")

st.code(
"""
streamlit_dashboard/
│
├── app.py
├── pages/
│   ├── 1_Data_Explorer.py
│   ├── 2_Iris_Predictor.py
│   └── 3_About.py
│
├── models/
│   └── dt_model.pkl
│
├── assets/
│   └── style.css
│
└── requirements.txt
"""
)


# ------------------------------------------------
# AUTHOR SECTION
# ------------------------------------------------
st.subheader("Author")

st.markdown(
    """
**Muhammed Faris T M**

Data Science & Analytics Enthusiast

This project was created as part of a **Data Science portfolio**
to demonstrate skills in:

- Data analytics
- Machine learning deployment
- Streamlit dashboard development
"""
)


# ------------------------------------------------
# FOOTER
# ------------------------------------------------
st.write("---")

st.caption(
    "Built with ❤️ using Streamlit | Data Analytics Dashboard"
)