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
stream_txt = "This section provides the complete overview of dataset to get going" \
    " along with much needed cleaning tools"
st.code(stream_txt)
uploaded_file = st.file_uploader("Upload your dataset to proceed", type=["csv"])
if uploaded_file is not None:
    st.success("File uploaded successfully!")
    df = pd.read_csv(uploaded_file)
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
        # Initialize session state dataframe
        # Initialize once
        if "clean_df" not in st.session_state or st.session_state.clean_df is None:
            st.session_state.clean_df = df.copy()

        on = st.toggle("Cleaning Tools")

        if on:
            st.code(
                "Drop or Replace NaN, Remove Duplicates and Unwanted Features, "
                "or Remove outliers to clean your dataset"
            )

            tab1, tab2, tab3, tab4 = st.tabs(["NaN", "Features", "Duplicates", "Outliers"])

            with tab1:
                st.subheader("Null Value Visualisation & Removal")
                col1, col2 = st.columns(2, border=True)
                with col1:
                    option1 = st.checkbox("Drop NaN")
                    option2 = st.checkbox("Replace NaN")

                    if option1:
                        result = data_cleaner.DropNaN(df)
                        if result is not None:
                            st.session_state.clean_df = result

                    if option2:
                        result = data_cleaner.ReplaceNaN(df)
                        if result is not None:
                            st.session_state.clean_df = result

                with col2:
                    st.write("Overview")
                    data_graphs.viewNaN_BarC(st.session_state.clean_df)
                    data_graphs.viewNaN_Pie(st.session_state.clean_df)

            with tab2:
                result = data_cleaner.DropFeature(st.session_state.clean_df)
                if result is not None:
                    st.session_state.clean_df = result

            with tab3:
                result = data_cleaner.DropDuplicates(st.session_state.clean_df)
                if result is not None:
                    st.session_state.clean_df = result

            with tab4:
                selc = st.segmented_control(
                    "Select Columns to view outliers",
                    st.session_state.clean_df.columns,
                    selection_mode="single",
                    default=st.session_state.clean_df.columns[0]
                )
                fig = px.box(
                    st.session_state.clean_df,
                    y=selc,
                    points="all",
                    title=f"Boxplot of {selc} with Outliers",
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)

                if pd.api.types.is_numeric_dtype(st.session_state.clean_df[selc]):
                    result = data_cleaner.RemoveOutliers(st.session_state.clean_df, selc)
                    if result is not None:
                        st.session_state.clean_df = result


        # You can now use st.session_state.clean_df anywhere after this block


    with st.container():
        st.subheader("Learn about your dataset with deepdive")
    # Always start from session state
        if "clean_df" not in st.session_state or st.session_state.clean_df is None:
            st.session_state.clean_df = df.copy()

        df1 = st.session_state.clean_df.copy()

        stream_txt = (
            "This section provides utilities for studying various elements "
            "under every feature present in the dataset"
        )
        st.code(stream_txt)

        selc = st.segmented_control(
            "Select A Column:",
            df1.columns,
            selection_mode="single",
            default=df.columns[0],
            key="subjective"
        )

        opt = st.multiselect(
            "Select An Action:",
            ["Search", "Value Count", "Selective view"],
            default="Search"
        )

        if "Search" in opt:
            result = data_subj.search(df1, selc)
            if result is not None:
                df1 = result

        if "Value Count" in opt:
            result = data_subj.ValueCount(df1, selc)
            if result is not None:
                df1 = result

        if "Selective view" in opt:
            result = data_subj.SelectView(df1, selc)
            if result is not None:
                df1 = result

        # Save back to session state so next container sees updates
        st.session_state.clean_df = df1
            