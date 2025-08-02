import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from src import data_models

df_model : pd.DataFrame

st.title("Model-Training")
st.header("Training Models based on Dataset")
col1,col2 = st.columns(2)
with col1:
    st.error("Please complete data encoding before model training.")
with col2:
    st.success("Or if you have a encoded dataset please upload")
if "dataframe" not in st.session_state:
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])
    if uploaded_file is not None:
        st.success("File uploaded successfully!")
        df_model = pd.read_csv(uploaded_file)
        st.session_state["dataframe"] = df_model
        st.subheader("Dataframe preview : ")
        with st.expander("view"):
            st.dataframe(df_model)
        
else:
    df_model = st.session_state["dataframe"]
    st.subheader("Dataframe preview : ")
    with st.expander("view"):
        st.dataframe(df_model)

if "dataframe" in st.session_state:
    st.subheader("Check Correlation:")
    corr_matrix = df_model.corr()
    fig = px.imshow(corr_matrix, text_auto=True, height=650, color_continuous_scale="RdBu", title="Correlation Matrix")
    with st.expander("View Correlation"):
        st.plotly_chart(fig, use_container_width=True)
        st.write("Correlation Insights:")
        st.code("""
near +1: indicates strong positive linear Relation
near -1: indicates strong negative linear Relation
near 0: indicates no linear Relation""")

    st.subheader("Dataset splitting and Training:")        
    st.code("""
How to select appropiate model for training:
1. observe the correlation matrix properly
2. if most of the values lies between -0.7 to 0.7 indicates classification models will be the appropiate choice
3. if majority of the values are close to 1 or -1 indicates Regression models will be the best choice""")
    st.markdown(''':green[However you will get access to almost all models along with proper performance metrics]''')
    st.markdown('''## :orange[Lets get started with selecting the Features and Target]''')

    st.markdown('''### :blue[Select the target & features:]''')
    col1, col2 = st.columns(2)
    with col1:
        features = st.multiselect("Select features (x)", df_model.columns, default=df_model.columns)
    with col2:
        target = st.selectbox("Select target (y)", df_model.drop(columns=features).columns)
        model_type = st.selectbox(
            "**Choose a model**",
            ["Linear Regression","Logistic Regression","KNN Regressor","Decision Tree Regressor","Decision Tree Classifier",
             "KNN Classifier","Naive_Bayes Classifier",
             "Support Vector Machine"]
        )
    st.markdown("▶ Features represent the :violet-background[independant variables] from which the model learns to predict the " \
    ":green-background[dependant variable 'target']")

    if target and features and model_type:
        X = df_model[features]
        y = df_model[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

        select = st.radio("Select Metrics type : ", ["Multivariate","Binary"], horizontal=True, help="select Multivalue when target consists of wide"
        " range of values for prediction and select Binary when target only consists of yes/no for prediction. !for linear models always select Multivalue")
        if 'trained_model' not in st.session_state:
            st.session_state.trained_model = None
            
        if st.button("Train Model"):
            model = data_models.train_model(X_train, y_train, model_type)
            st.session_state.trained_model = model
            
            if select == "Multivariate":
                data_models.evaluate_Multitarget_model(model, X_test, y_test)
            else:
                data_models.evaluate_binary_model(model, X_test, y_test)
        with st.expander("Test Predictions : "):
        # Add prediction section after model training
            if st.session_state.trained_model is not None:
                st.subheader("Make Predictions")
                has_encodings = 'encoders' in st.session_state
                
                input_data = {}
                cols = st.columns(2)
                for i, feature in enumerate(features):
                    with cols[i % 2]:
                        # For categorical features with known encodings
                        if has_encodings and feature in st.session_state.encoders:
                            # Get categories from the original encoder
                            encoder = st.session_state.encoders[feature]
                            # For OrdinalEncoder, we need to access categories_[0]
                            categories = list(encoder.categories_[0])
                            selected = st.selectbox(f"{feature}", options=categories)
                            # Transform the selected value
                            input_data[feature] = encoder.transform([[selected]])[0][0]
                        else:
                            # Default numeric input
                            min_val = float(X_train[feature].min())
                            max_val = float(X_train[feature].max())
                            input_data[feature] = st.number_input(
                                f"{feature} (range: {min_val:.2f} to {max_val:.2f})",
                                min_value=min_val,
                                max_value=max_val,
                                value=(min_val + max_val)/2
                            )
                        
                if st.button("Predict", key="prediction_button"):
                    input_df = pd.DataFrame([input_data])
                    prediction = st.session_state.trained_model.predict(input_df)
                    
                    # Check if target was encoded and decode if needed
                    if has_encodings and target in st.session_state.encoders:
                        target_encoder = st.session_state.encoders[target]
                        # For classification, use inverse_transform
                        if select == "Binary" or select == "Multivariate":
                            decoded_prediction = target_encoder.inverse_transform(prediction.reshape(-1, 1))[0][0]
                            st.success(f"Predicted {target}: {decoded_prediction}")
                        # For regression with encoded target (less common), you might need different handling
                        else:
                            st.success(f"Predicted {target}: {prediction[0]}")
                    else:
                        # Original prediction display logic
                        if select == "Binary":
                            res = True if prediction[0] == 1 else False
                            st.success(f"Predicted {target}: {res}")
                        else:
                            st.success(f"Predicted {target}: {prediction[0]}")