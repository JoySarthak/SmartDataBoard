import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from src import data_loader, data_subj, data_cleaner, data_graphs
from streamlit.runtime.scriptrunner import get_script_run_ctx

ctx = get_script_run_ctx()
if ctx and hasattr(ctx, "uploaded_file_mgr"):
    ctx.uploaded_file_mgr.max_upload_size = 500 * 1024 * 1024 
import random

def Generate_key():
    num = random.random()
    return str(num)

st.title("Exploratory Data Analysis")
st.header("Data Overview")
uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])
if uploaded_file is not None:
    st.success("File uploaded successfully!")
    df = pd.read_csv(uploaded_file)
    stream_txt = "This section provides with the complete overview of dataset to get going" \
    " along with much needed cleaning tools"
    st.code(stream_txt)
    data_loader.Oversight(df)
    with st.expander("View Options"):
        sclt1 = st.selectbox("Choose what to view", ["Dataframe","Description","Info Overview"])
        if sclt1 == "Dataframe":
            data_loader.overview(df,"1")
        if sclt1 == "Description":
            data_loader.description(df)
        if sclt1 == "Info Overview":
            data_loader.general_info(df)
            data_graphs.viewNaN_line(df)
                            
    with st.container():
        on = st.toggle("Cleaning Tools")
        if on:
            Copy_df = df.copy()
            st.code("Drop or Replace NaN, Remove Duplicates and Unwanted Features, "
            "or Remove outliers to clean your dataset")
            tab1, tab2, tab3, tab4 = st.tabs(["NaN", "features", "Duplicates", "Outliers"])
            with tab1:
                st.subheader("Null Value Visualisation & Removal")
                col1,col2 = st.columns(2, border=True)
                with col1:
                    option1 = st.checkbox("Drop NaN")
                    option2 = st.checkbox("Replace NaN")
                    
                    if option1:
                        Copy_df = data_cleaner.DropNaN(Copy_df)
                    if option2:
                        Copy_df = data_cleaner.ReplaceNaN(Copy_df)

                with col2:
                    st.write("Overview")
                    if Copy_df is not None:
                        data_graphs.viewNaN_BarC(Copy_df)
                        data_graphs.viewNaN_Pie(Copy_df)
                    else:
                        data_graphs.viewNaN_BarC(df)
                        data_graphs.viewNaN_Pie(df)

            with tab2:
                if Copy_df is not None:
                    Copy_df = data_cleaner.DropFeature(Copy_df)
                else:
                    Copy_df = data_cleaner.DropFeature(df)

            with tab3:
                Copy_df = data_cleaner.DropDuplicates(Copy_df)
            
            with tab4:
                selc = st.segmented_control("Select Columns to view outliers",df.columns,selection_mode="single", default=df.columns[0])
                fig = px.box(df, y=selc, points="all", title=f"Boxplot of {selc} with Outliers", height=600)
                st.plotly_chart(fig, use_container_width=True)
                if pd.api.types.is_numeric_dtype(df[selc]):
                    data_cleaner.RemoveOutliers(df, selc)