import time
import streamlit as st
import pandas as pd
from src import data_loader
from Smart_tool import smart_train

txt_msg = """The Smart Model uses AI to automatically check all the relative models suitable for your dataset
once you upload and select your target variable, sit back relax and your report will be ready."""

st.header("Smart Model Training")
st.markdown(":orange[Tired of checking all ML models manually?]:green[ Why not try out the Smart Model Training]")

def stream_data():
    for word in txt_msg.split(" "):
        yield word + " "
        time.sleep(0.05)

st.write_stream(stream_data)

uploaded_file = st.file_uploader("upload a csv")
if uploaded_file is not None:
    st.success("file uploaded successfully")
    df = pd.read_csv(uploaded_file)
    st.subheader("View general info: ")
    with st.expander("view"):
        data_loader.general_info(df)
        st.error("if your dataset contains null values please use the cleaning tools available under EDA-overview to avoid errors")

    target = st.selectbox("Select target", df.columns, index=None)
    if target is not None:
        if st.button("smart train"):
            smart_train.smart_training(df, target)