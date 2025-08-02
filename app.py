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
        st.Page("pages/overview_data.py", title="Overview"),
        st.Page("pages/subjective_data.py", title="Subjective Analysis"),
        st.Page("pages/visualise_data.py", title="Visualise Data")
    ],
    "Model Training": [
        st.Page("pages/ML_data_encoding.py", title="Data Encoding"),
        st.Page("pages/ML_models.py", title="Data Training"),
        st.Page("pages/project.py", title="Prediction Model")
    ]
}

pg = st.navigation(pages, position="top")
pg.run()
