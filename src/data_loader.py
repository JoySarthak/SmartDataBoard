import streamlit as st
import pandas as pd
def Oversight(df : pd.DataFrame):
    No_of_col = len(df.columns)
    No_of_unique = len(df)
    No_of_obj = (df.dtypes == 'object').sum()
    No_of_numb = (df.dtypes != 'object').sum()
    N_Col,N_Unique,N_obj,N_num = st.columns(4, gap='medium')
    with N_Col:
        st.info('features')
        with st.container(border=True):
            st.metric(label='Number of Columns',value=f"{No_of_col}")
    with N_Unique:
        st.info('Unique Values')
        with st.container(border=True):
            st.metric(label='Number of Values',value=f"{No_of_unique}")
    with N_obj:
        st.info('Objects')
        with st.container(border=True):
            st.metric(label='Number of Object columns',value=f"{No_of_obj}")
    with N_num:
        st.info('Numeric')
        with st.container(border=True):
            st.metric(label='Number of Numeric Columns',value=f"{No_of_numb}")

def overview(df : pd.DataFrame, key_suffix: str = ""):
    max_rows = len(df) 
    showdata = st.multiselect(
    'Show Column: ',
    df.columns,
    default=list(df.columns),  # This sets all columns as selected by default
    placeholder="Select columns to view",
    key=f"{key_suffix}"
)
    no_of_rows = st.slider("Number of rows", 10, max_rows, 100)
    st.write(df[showdata].head(no_of_rows))

def description(df : pd.DataFrame):
    st.dataframe(df.describe())
    st.subheader("Graphical Overview")
    with st.container(border=True):
        st.line_chart(df.describe(), width=None, height=550, use_container_width=True)

def general_info(df : pd.DataFrame):
    summary = pd.DataFrame({
        'features': df.columns,
        'Non-Null Values': df.count().values,
        'Null Values': df.isna().sum().values,
        'dtype': [str(x) for x in df.dtypes]  # Force string conversion
    })
    st.dataframe(summary)

def Downloaddf(df: pd.DataFrame, key_suffix: str = ""):
    st.download_button(
        label="Download cleaned CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name="cleaned_data.csv",
        mime="text/csv",
        icon=":material/download:",
        key=f"download_button_{key_suffix}",
        type="primary"
    )