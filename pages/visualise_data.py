import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from src import data_loader, data_subj, data_cleaner, data_graphs

st.title("Exploratory Data Analysis")
st.header("Visualize Data Through Charts")
uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])
if uploaded_file is not None:
    st.success("File uploaded successfully!")
    df = pd.read_csv(uploaded_file)
    stream_txt = "This section provides visuals of data trends through charts" \
    " and select theme option too."
    st.code(stream_txt)
    st.write("Appearance")
    color = data_graphs.color_scheme_selector("overall")
    st.subheader("Visualise Counts") 
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            sclt_graph = st.segmented_control("Choose Chart Type:",["Bar","line","Area","Pie"],
                                            selection_mode="single",default="Bar")
        with col2:
            opt = st.selectbox("Select a feature:",df.columns, key="Valc")
        if "Bar" in sclt_graph:
            data_graphs.Graph_Bar(df, "Value Counts", opt, color)
        if "line" in sclt_graph:
            data_graphs.Graph_line(df, "Value Counts", opt, color)
        if "Area" in sclt_graph:
            data_graphs.Graph_area(df, "Value Counts", opt, color)
        if "Pie" in sclt_graph:
            data_graphs.Graph_pie(df, opt, "Value Counts", color)

    st.subheader("Visualise with Search")
    with st.container(border=True):
        colx, coly = st.columns(2)
        with colx:
            sclt_graph = st.segmented_control("Choose Chart Type:",["Bar","line","Area","Pie"],
                                            selection_mode="single",default="Bar",key="V_search")
        with coly:
            opt = st.selectbox("Select a feature:",df.columns, key="Search")
        if "Bar" in sclt_graph:
            data_graphs.Graph_Bar(df, "Searc", opt, color)
        if "line" in sclt_graph:
            data_graphs.Graph_line(df, "Searc", opt, color)
        if "Area" in sclt_graph:
            data_graphs.Graph_area(df, "Searc", opt, color)
        if "Pie" in sclt_graph:
            data_graphs.Graph_pie(df, opt, "Searc", color)

    st.subheader("Visualise Trends")
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            sclt_graph = st.segmented_control("Choose Chart Type:",["Bar","line","Area"],
                                            selection_mode="single",default="Bar")
        with col2:
            opt = st.selectbox("Select a feature:",df.columns,key="AGG")
        if "Bar" in sclt_graph:
            data_graphs.Graph_Bar(df, "Agg", opt, color)
        if "line" in sclt_graph:
            data_graphs.Graph_line(df, "Agg", opt, color)
        if "Area" in sclt_graph:
            data_graphs.Graph_area(df, "Agg", opt, color)


