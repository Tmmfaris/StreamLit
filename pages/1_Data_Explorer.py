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

```
df = pd.DataFrame(data)
df["Profit"] = df["Sales"] - df["Expenses"]

return df
```

df = load_data()

# ------------------------------------------------

# DATASET METRICS

# ------------------------------------------------

st.subheader("Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Average Sales", f"${df['Sales'].mean():.0f}")
col4.metric("Total Profit", f"${df['Profit'].sum():.0f}")

# ------------------------------------------------

# FILTERS

# ------------------------------------------------

st.subheader("Filters")

col1, col2 = st.columns(2)

city_filter = col1.selectbox(
"Select City",
["All"] + sorted(df["City"].unique().tolist())
)

age_filter = col2.slider(
"Select Age Range",
min_value=int(df["Age"].min()),
max_value=int(df["Age"].max()),
value=(int(df["Age"].min()), int(df["Age"].max()))
)

# ------------------------------------------------

# APPLY FILTERS

# ------------------------------------------------

filtered_df = df.copy()

if city_filter != "All":
filtered_df = filtered_df[filtered_df["City"] == city_filter]

filtered_df = filtered_df[
(filtered_df["Age"] >= age_filter[0]) &
(filtered_df["Age"] <= age_filter[1])
]

# ------------------------------------------------

# DATA TABLE

# ------------------------------------------------

st.subheader("Dataset Table")

st.data_editor(
filtered_df,
use_container_width=True
)

# ------------------------------------------------

# SUMMARY STATISTICS

# ------------------------------------------------

with st.expander("Summary Statistics"):

```
st.dataframe(
    filtered_df.describe(),
    use_container_width=True
)
```

# ------------------------------------------------

# CHARTS

# ------------------------------------------------

st.subheader("Charts")

col1, col2 = st.columns(2)

with col1:
st.markdown("Sales by Person")
st.bar_chart(
filtered_df.set_index("Name")["Sales"]
)

with col2:
st.markdown("Profit by Person")
st.bar_chart(
filtered_df.set_index("Name")["Profit"]
)

# ------------------------------------------------

# SALES VS EXPENSES

# ------------------------------------------------

st.subheader("Sales vs Expenses")

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
file_name="filtered_dataset.csv",
mime="text/csv"
)

# ------------------------------------------------

# ABOUT SECTION

# ------------------------------------------------

with st.expander("About this page"):

```
st.markdown(
    """
    This page allows users to:

    • Explore dataset tables  
    • Filter data by city and age  
    • Visualize sales and profit  
    • View statistical summaries  
    • Download filtered data  

    These features simulate **real analytics dashboards used in business intelligence tools**.
    """
)
```
