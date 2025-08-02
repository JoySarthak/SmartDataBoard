import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from src import data_loader, data_subj, data_cleaner, data_graphs



st.title("Exploratory Data Analysis")
st.header("Subjective Analysis")
uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])
if uploaded_file is not None:
    st.success("File uploaded successfully!")
    df = pd.read_csv(uploaded_file)
    stream_txt = "This section provides utilities for studying various elements under every features" \
    " present in the dataset"
    st.code(stream_txt)
    selc = st.segmented_control("Select A Column:",df.columns,selection_mode="single", default=df.columns[0])
    opt = st.multiselect("Select An Action: ",["Search","Value Count","Groupby","Selective view","Replace"], default="Search")
    if "Search" in opt:
        data_subj.search(df,selc)
    if "Value Count" in opt:
        data_subj.ValueCount(df, selc)
    if "Selective view" in opt:
        data_subj.SelectView(df, selc)
    if "Replace" in opt:
        data_subj.ReplaceVal(df, selc)
    if "Groupby" in opt:
        data_subj.Grouping(df, selc)