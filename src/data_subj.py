import streamlit as st
import pandas as pd
import numpy as np
def value_convt(df, column_name, search_value):
    try:
        col_dtype = df[column_name].dtype
        
        # Convert the input value to match the column's dtype
        if col_dtype == 'int64':
            converted_value = int(search_value)
        elif col_dtype == 'float64':
            converted_value = float(search_value)
        elif col_dtype == 'bool':
            converted_value = search_value.lower() in ['true', '1', 'yes']
        elif np.issubdtype(col_dtype, np.datetime64):
            converted_value = pd.to_datetime(search_value)
        else:
            converted_value = str(search_value)  # fallback to string
        
        # Perform the search
        return converted_value
    
    except ValueError:
        raise ValueError(f"Could not convert '{search_value}' to match the column's data type ({col_dtype})")

def search(df : pd.DataFrame, sclt_colmn):
    with st.expander("Searching tools"):
        search_value = st.text_input("Enter the value to search")
        if search_value:
            search_value = value_convt(df,sclt_colmn,search_value)
            sdf = df[df[sclt_colmn] == search_value]
            st.dataframe(sdf)
            with st.expander("More Option"):
                col = st.selectbox("Find value counts with respect to:", df.columns)
                if col and search_value:
                    st.dataframe(df[df[sclt_colmn] == search_value][col].value_counts().reset_index())
        if df[sclt_colmn].dtype.kind in 'iufc':
            st.subheader("For numeric dtype columns also search by :")
            operation = st.selectbox("Select operation", 
                                    ["!=", ">", "<", ">=", "<=", "between"])
            val1 = st.number_input("Enter the value:",min_value=0)
            if operation == "between":
                val2 = st.number_input("Enter the second value:",min_value=0,placeholder="if using between")
            if st.button("Apply Filter"):
                try:
                    if operation == "!=":
                        result = df[df[sclt_colmn] != val1]
                    elif operation == ">":
                        result = df[df[sclt_colmn] > val1]
                    elif operation == "<":
                        result = df[df[sclt_colmn] < val1]
                    elif operation == ">=":
                        result = df[df[sclt_colmn] >= val1]
                    elif operation == "<=":
                        result = df[df[sclt_colmn] <= val1]
                    elif operation == "between":
                        result = df[(df[sclt_colmn]>val1) & (df[sclt_colmn]<val2)]
                    st.write("Filtered Results:")
                    st.dataframe(result)
                except Exception as e:
                    st.error(f"Error applying filter: {e}")       


def ValueCount(df, sclt_colmn):
    with st.expander("Value count tools"):
        st.subheader(f"Value Counts Of Elements in {sclt_colmn}")
        st.dataframe(df[sclt_colmn].value_counts().reset_index())

def callagain(df, scl, counter):
    SelectView(df, scl, str(counter))  # Convert counter to string for key suffix

def SelectView(df, sclt_colmn, key_suffix: str = "1"):
    with st.expander("Customise views"):
        option = df[sclt_colmn].unique()
        selected_options = st.multiselect(
            f"Choose what to view",
            option,
            key=f"multiselect_{key_suffix}"  # Add prefix to make key unique
        )
        if selected_options:
            filtered_df = df[df[sclt_colmn].isin(selected_options)]
            st.dataframe(filtered_df)
            scl2 = st.segmented_control("View against: ",df.columns, selection_mode="single", 
                                        default=df.columns[0], key=f"selectbox_{key_suffix}")
            callagain(filtered_df, scl2, int(key_suffix) + 1)  # Increment counter for next call
        else:
            st.dataframe(df)

def ReplaceVal(df, sclt_colmn):
    with st.expander("Replace any value in column", expanded=True):
        r_val = st.text_input("Enter the value to be replaced")
        new_val = st.text_input("Enter the new value to replace")
        if r_val:
            r_val = value_convt(df,sclt_colmn,r_val)
            st.subheader("Rows indentified for change : ")
            st.dataframe(df[df[sclt_colmn] == r_val])
        if r_val and new_val:
            r_val = value_convt(df,sclt_colmn,r_val)
            new_val = value_convt(df,sclt_colmn,new_val)
            
        if st.button("Replace"):
            sdf = df.copy()

            # Perform the replacement
            sdf[sclt_colmn] = sdf[sclt_colmn].replace(r_val, new_val)

            # Find rows where the value was changed
            changed_rows = df[sclt_colmn] != sdf[sclt_colmn]

            if changed_rows.any():
                st.subheader(f"Rows updated after Change : ")
                st.dataframe(sdf[changed_rows])  # ✅ Show full rows where replacement happened
            else:
                st.info("No values were replaced.")
            return sdf

def Grouping(df, sclt_colmn):
    with st.expander("Groupby Insights"):
        st.subheader(f"Number of values in every column w.r.t {sclt_colmn}")
        st.dataframe(df.groupby(sclt_colmn).nunique())

