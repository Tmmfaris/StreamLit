import streamlit as st
import pandas as pd


# ------------------------------------------------
# PAGE TITLE
# ------------------------------------------------
st.title("Data Explorer")
st.write("Explore and analyze the sample dataset interactively.")


# ------------------------------------------------
# LOAD SAMPLE DATA
# ------------------------------------------------
@st.cache_data
def load_data():
    data = {
        "Name": ["Alice", "Bob", "Charlie", "David", "Emma"],
        "Age": [25, 30, 35, 28, 22],
        "City": ["New York", "Los Angeles", "Chicago", "Houston", "Miami"],
        "Sales": [150, 200, 250, 180, 120],
        "Expenses": [100, 150, 200, 130, 90],
    }

    df = pd.DataFrame(data)
    df["Profit"] = df["Sales"] - df["Expenses"]
    return df


df = load_data()


# ------------------------------------------------
# DATASET OVERVIEW
# ------------------------------------------------
st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Average Sales", f"${df['Sales'].mean():.0f}")


# ------------------------------------------------
# DATA FILTERS
# ------------------------------------------------
st.subheader("Filters")

city_filter = st.selectbox(
    "Select City",
    options=["All"] + sorted(df["City"].unique().tolist())
)

if city_filter != "All":
    filtered_df = df[df["City"] == city_filter]
else:
    filtered_df = df


# ------------------------------------------------
# DATA TABLE
# ------------------------------------------------
st.subheader("Dataset Table")

st.data_editor(filtered_df, use_container_width=True)


# ------------------------------------------------
# STATISTICS
# ------------------------------------------------
with st.expander("Summary Statistics"):

    st.dataframe(filtered_df.describe(), use_container_width=True)


# ------------------------------------------------
# CHARTS
# ------------------------------------------------
st.subheader("Charts")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Sales Distribution")
    st.bar_chart(filtered_df.set_index("Name")["Sales"])

with col2:
    st.markdown("#### Profit Comparison")
    st.bar_chart(filtered_df.set_index("Name")["Profit"])


# ------------------------------------------------
# LINE CHART
# ------------------------------------------------
st.markdown("### Sales vs Expenses")

chart_data = filtered_df[["Sales", "Expenses"]]
st.line_chart(chart_data)


# ------------------------------------------------
# DOWNLOAD DATA
# ------------------------------------------------
st.subheader("Download Data")

csv = filtered_df.to_csv(index=False)

st.download_button(
    label="Download CSV",
    data=csv,
    file_name="dataset.csv",
    mime="text/csv",
)


# ------------------------------------------------
# INFO SECTION
# ------------------------------------------------
with st.expander("About this page"):

    st.markdown(
        """
        This page allows users to:

        - Explore dataset tables
        - Filter data by city
        - Visualize sales and profit
        - View statistical summaries
        - Download filtered data

        This functionality is commonly used in **data analytics dashboards**.
        """
    )