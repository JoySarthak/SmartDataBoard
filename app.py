import streamlit as st

st.set_page_config(
    page_icon="data/logo2.png",
    page_title="Smartboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.image("data/photo1.jpeg", caption="Smart-Data-board")

pages = {
    "Data Analysis": [
        st.Page("overview_data.py", title="Overview"),
        st.Page("subjective_data.py", title="Subjective Analysis"),
        st.Page("visualise_data.py", title="Visualise Data")
    ],
    "Model Training": [
        st.Page("ML_data_encoding.py", title="Data Encoding"),
        st.Page("ML_models.py", title="Data Training"),
        st.Page("project.py", title="Prediction Model")
    ]
}

pg = st.navigation(pages, position="top")
pg.run()
