import streamlit as st
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import pickle  # Add this import

st.title("Model-Training")
st.header("Data Encoding")

st.write("For machine learning models to perform optimally Data encoding is a crucial process" \
" please complete and download the encoded dataset before proceeding to the Data training section")

st.warning("⚠️ if your dataset contains null values or redundant elements please perform a cleanup before proceeding ⚠️", width=800)

uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])
if uploaded_file is not None:
    st.success("File uploaded successfully!")
    df = pd.read_csv(uploaded_file)
    st.session_state["dataframe"] = df

    if st.button("Encode Categorical Data", type="secondary"):
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        # Create and store encoders for each column
        encoders = {}
        for col in categorical_columns:
            encoder = OrdinalEncoder()
            df[col] = encoder.fit_transform(df[[col]])
            encoders[col] = encoder
        
        # Store encoders in session state
        st.session_state["encoders"] = encoders
        st.session_state["dataframe"] = df
        
        # Also save encoders to file for download
        with open("encoders.pkl", "wb") as f:
            pickle.dump(encoders, f)
    
    st.subheader("Dataframe view: ")
    df = st.session_state["dataframe"]
    st.dataframe(df)
    
    col1, col2, col3 = st.columns([0.25, 0.55, 0.2])  # Added column for encoder download
    with col1:
        st.download_button(
            label="Download encoded CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name="encoded.csv",
            mime="text/csv",
            icon=":material/download:",
            key="encoded",
            type="primary"
        )
    with col2:
        if "encoders" in st.session_state:
            st.download_button(
                label="Download Encoders",
                data=open("encoders.pkl", "rb"),
                icon=":material/download:",
                file_name="encoders.pkl",
                mime="application/octet-stream",
                type="primary"
            )
    with col3:
        st.page_link("pages/ML_models.py", label="Goto Model Training ➡")