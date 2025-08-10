import streamlit as st
import pandas as pd

st.title("Search, Replace and Customize data")
st.write("Search values from dataset, columns and replace them with ease, contains various other tools")
uploaded_file = st.file_uploader("Upload your dataset to proceed", type=["csv"])
if uploaded_file is not None:
    st.success("File uploaded successfully!")
    data = pd.read_csv(uploaded_file)
    st.subheader(":blue[Quick edit: Find and Replace]")
    if st.toggle("Enable editing"):
        edited_data = st.data_editor(data, use_container_width=True)
        st.download_button(
        label="Download edited csv",
        data=edited_data.to_csv(index=False).encode('utf-8'),
        file_name="data_processed.csv",
        mime="text/csv",
        icon=":material/download:",
        key=f"Edit_download",
        type="primary"
    )
    else:
        st.dataframe(data, use_container_width=True)
    
    st.subheader(":green[Rename Columns]")
    new_names = {}
    st.write("##### Enter new names for the columns:")
    # Display each column with a text input beside it
    st.markdown("""
    <style>
    .label-box {
        font-size: 22px;
        height: 38px; 
        line-height: 38px; 
        font-family: monospace;
        background-color: #262730;
        padding: 0 8px;
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)
    for col in data.columns:
        c1, c2 = st.columns([0.3, 0.6])
        c1.markdown(f"<div class='label-box'> {col} : </div>", unsafe_allow_html=True)
        new_name = c2.text_input(
            "Enter the new name:",
            value=col,
            key=f"rename_{col}",
            label_visibility="collapsed",
            placeholder="Enter the new name"
        )
        new_names[col] = new_name
    # Button to apply renaming
    if st.button("Apply", type="primary"):
        data.rename(columns=new_names, inplace=True)
        st.success("Columns renamed successfully!")
        st.dataframe(data)
        st.download_button(
        label="Download edited csv",
        data=data.to_csv(index=False).encode('utf-8'),
        file_name="data_processed.csv",
        mime="text/csv",
        icon=":material/download:",
        key=f"Edit_download2",
        type="primary"
    )
    
