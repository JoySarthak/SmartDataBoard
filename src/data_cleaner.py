import streamlit as st
import pandas as pd
from src import data_loader
import plotly.express as px

def DropNaN(df : pd.DataFrame):
    Cdf = df.dropna()
    st.dataframe(Cdf.isna().sum().reset_index()
                 .rename(columns={'index': 'Attributes', 0: 'Missing Value Count'}))
    st.write(f"Number of Rows Removed : {len(df)-len(Cdf)}")
    return Cdf

def ReplaceNaN(df : pd.DataFrame):
    
    showdata = st.multiselect('Select Column : ',df.columns,default=[],
                              placeholder="Select columns to Replace NaN")
    data_loader.general_info(df[showdata])
    selected_items = {}
    df_cleaned = df.copy()
    st.subheader("Handle Missing Values:")
    with st.expander("Options"):
        for item in showdata:
            if st.checkbox(item, key=f"check_{item}"):
                col_dtype = df[item].dtype
                # Options for how to fill missing values
                option = st.segmented_control(
                f"Select method for {item}:", 
                ["Mean", "Median", "Mode", "Custom Value"], 
                key=f"radio_{item}",
                selection_mode="single"  # This makes the radio buttons display horizontally
                )

            # If custom, show input box for value (string or numeric)
                custom_value = None
                if option == "Custom Value":
                    custom_value = st.text_input(f"Enter custom value for {item}:", key=f"custom_{item}")

                selected_items[item] = {
                'method': option,
                'custom_value': custom_value,
                'dtype': col_dtype
                }
        
        if st.button("Apply Missing Value Handling", type="primary"):
            for item, details in selected_items.items():
                if details['method'] == "Mean":
                    if pd.api.types.is_numeric_dtype(details['dtype']):
                        df_cleaned[item] = df_cleaned[item].fillna(df_cleaned[item].mean())
                    else:
                        st.warning(f"⚠️ Cannot apply mean to non-numeric column '{item}'")

                elif details['method'] == "Mode":
                    mode_val = df_cleaned[item].mode()
                    if not mode_val.empty:
                        df_cleaned[item] = df_cleaned[item].fillna(mode_val.iloc[0])
                    else:
                        st.warning(f"⚠️ No mode found for column '{item}'")

                elif details['method'] == "Median":
                    if pd.api.types.is_numeric_dtype(details['dtype']):
                        df_cleaned[item] = df_cleaned[item].fillna(df_cleaned[item].median())
                    else:
                        st.warning(f"⚠️ Cannot apply median to non-numeric column '{item}'")
                elif details['method'] == "Custom Value":
                # Convert to numeric if column is numeric and user input is numeric
                    if pd.api.types.is_numeric_dtype(details['dtype']):
                        try:
                            fill_value = float(details['custom_value'])
                        except:
                            fill_value = details['custom_value']  # fallback to string if conversion fails
                    else:
                        fill_value = details['custom_value']
                        df_cleaned[item] = df_cleaned[item].fillna(fill_value)
            st.subheader("Results: ")
            data_loader.general_info(df_cleaned[showdata])
            data_loader.Downloaddf(df_cleaned,"1")
            return df_cleaned

def DropFeature(df : pd.DataFrame):
    df_drop = df.copy()
    drop_select = st.segmented_control("Select Columns to Drop",df.columns,selection_mode="multi")
    df_drop = df.drop(columns=drop_select)
    st.write(df_drop)
    data_loader.Downloaddf(df_drop,"2")
    return df_drop

def DropDuplicates(df : pd.DataFrame):
    n_dup = df.duplicated().sum()
    st.code(f"Number of Duplicates found in Dataset: {n_dup}")
    if st.button("Remove Duplicates",type="primary"):
        df = df.drop_duplicates()
        st.code(f"Removed all duplicates")
        data_loader.Downloaddf(df,"3")
    return df

def RemoveOutliers(df : pd.DataFrame, selc):
        with st.expander("view outlier removal options for numeric columns"):
            Q1 = df[selc].quantile(0.15)
            Q3 = df[selc].quantile(0.85)
            IQR = Q3 - Q1
            outliers = df[(df[selc] < Q1 - 1.5 * IQR) | (df[selc] > Q3 + 1.5 * IQR)]
            st.subheader(f"Outliers in {selc} are: ")
            st.write(outliers)
            st.warning("⚠️Careful! Outliers can provide valuable insights on Data distribution, removing" \
            " them might cause loss in values which might be necessary for proper analysis.")
            if st.button("Remove Outliers"):
                filtered_df = df[(df[selc] >= Q1 - 1.5 * IQR) & (df[selc] <= Q3 + 1.5 * IQR)]
                fig = px.box(filtered_df, y=selc, points="all", title=f"Boxplot of {selc} without Outliers", height=600)
                st.plotly_chart(fig, use_container_width=True)
                data_loader.Downloaddf(filtered_df,"4")