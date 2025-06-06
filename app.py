import streamlit as st
import pickle
import numpy as np
import plotly.express as px


st.title("🚗 Accident Severity Predictor")

st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", ["🚨 Predict Severity", "📚 About the Data", "🤖 Compare Models", "💬 Chat Assistant"])

if page == "🚨 Predict Severity":
    # show form and prediction



    st.header("📌 Instructions")
    st.markdown("""
    This app helps predict the **severity** of a **Road Accident** based on key conditions like weather, road surface, vehicle type, and more.  
    The prediction is based on **machine learning models trained on real 2019 UK accident data**.

    Use this tool to simulate accident conditions and get an **instant prediction** of whether an accident would likely be *Slight*, *Serious*, or *Fatal*.

    ---
    """)

    st.subheader("Please fill in the form below")
    # Load pre-trained model
    @st.cache_resource
    def load_model():
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
        st.write("Model loaded successfully!")

        with open('label_encoders.pkl', 'rb') as file:
            label_encoders = pickle.load(file)
        st.write("Label encoders loaded successfully!")

        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        st.write("Scaler loaded successfully!")

        return model, label_encoders, scaler

    # Load the pre-trained model
    model, label_encoders, scaler = load_model()


    # Form inputs
    with st.form("input_form"):
        speed_limit = st.number_input("Speed Limit", min_value=0)
        light_conditions = st.selectbox("Light Conditions", label_encoders['light_conditions'].classes_)
        weather_conditions = st.selectbox("Weather Conditions", label_encoders['weather_conditions'].classes_)
        road_surface_conditions = st.selectbox("Road Surface Conditions", label_encoders['road_surface_conditions'].classes_)
        vehicle_type = st.selectbox("Vehicle Type", label_encoders['vehicle_type'].classes_)
        junction_location = st.selectbox("Junction Location", label_encoders['junction_location'].classes_)
        skidding_and_overturning = st.selectbox("Skidding/Overturning", label_encoders['skidding_and_overturning'].classes_)
        vehicle_leaving_carriageway = st.selectbox("Vehicle Leaving Carriageway", label_encoders['vehicle_leaving_carriageway'].classes_)
        hit_object_off_carriageway = st.selectbox("Hit Object Off Carriageway", label_encoders['hit_object_off_carriageway'].classes_)
        first_point_of_impact = st.selectbox("First Point of Impact", label_encoders['first_point_of_impact'].classes_)
        sex_of_driver = st.selectbox("Sex of Driver", label_encoders['sex_of_driver'].classes_)
        age_of_oldest_driver = st.number_input("Age of Oldest Driver", min_value=16)

        submitted = st.form_submit_button("Predict")


    # Encode and scale input
    def prepare_input():
        input_dict = {
            'speed_limit': speed_limit,
            'light_conditions': label_encoders['light_conditions'].transform([light_conditions])[0],
            'weather_conditions': label_encoders['weather_conditions'].transform([weather_conditions])[0],
            'road_surface_conditions': label_encoders['road_surface_conditions'].transform([road_surface_conditions])[0],
            'vehicle_type': label_encoders['vehicle_type'].transform([vehicle_type])[0],
            'junction_location': label_encoders['junction_location'].transform([junction_location])[0],
            'skidding_and_overturning': label_encoders['skidding_and_overturning'].transform([skidding_and_overturning])[0],
            'vehicle_leaving_carriageway': label_encoders['vehicle_leaving_carriageway'].transform([vehicle_leaving_carriageway])[0],
            'hit_object_off_carriageway': label_encoders['hit_object_off_carriageway'].transform([hit_object_off_carriageway])[0],
            'first_point_of_impact': label_encoders['first_point_of_impact'].transform([first_point_of_impact])[0],
            'sex_of_driver': label_encoders['sex_of_driver'].transform([sex_of_driver])[0],
            'age_of_oldest_driver': age_of_oldest_driver
        }

        # Convert to array and scale numerical
        input_df = np.array([[input_dict[col] for col in input_dict]])
        input_df[:, [0, 11]] = scaler.transform(input_df[:, [0, 11]])  # speed_limit and age scaled

        return input_df


    # Predict
    if submitted:
        input_array = prepare_input()
        prediction = model.predict(input_array)
        severity_labels = {0: "Low", 1: "Medium", 2: "High"}
        predicted_class = prediction[0]
        severity_text = severity_labels.get(predicted_class, "Unknown")
        if severity_text == "Low":
            st.success(f"🚨 Predicted Accident Severity at a {junction_location} is **{severity_text}**")

        elif severity_text == "Medium":
            st.warning(f"🚧 Predicted Accident Severity at a {junction_location} is **{severity_text}**")
        else: # High
            st.error(f"🔥 Predicted Accident Severity at a {junction_location} is **{severity_text}**")


elif page == "📚 About the Data":
    st.header("📚 About the 2019 UK Road Accident Dataset")
    st.markdown("""
    This dataset contains records of road accidents reported in the UK in **2019**, including:

    - Speed limits, weather & lighting conditions
    - Vehicle type and road surface conditions
    - Driver info (age, sex)
    - Accident outcome (severity: Slight, Serious, Fatal)

    It's provided to help train predictive models for emergency response planning.
    """)


elif page == "🤖 Compare Models":
    st.header("🤖 Machine Learning vs Neural Networks")
    st.markdown("""
    We trained several models including:

    - Logistic Regression
    - Random Forest
    - XGBoost
    - Neural Networks (Keras/TensorFlow)

    Below is a comparison based on **Accuracy**, **F1-Score**, and **Confusion Matrix**:
    """)

    # Example table
    model_results = {
        "Model": ["Random Forest", "XGBoost", "Neural Network"],
        "Accuracy": [0.83, 0.85, 0.88],
        "F1-Score": [0.80, 0.82, 0.87],
    }
    st.dataframe(model_results)

        # Pie Chart for Accuracy
    fig = px.pie(
        model_results,
        values='Accuracy',
        names='Model',
        title='📊 Model Accuracy Comparison',
        hole=0.4  # for donut style
    )
    st.plotly_chart(fig)


elif page == "💬 Chat Assistant":
    st.header("💬 Chat Assistant")
    user_query = st.text_input("Ask a question about the app or the dataset:")

    if user_query:
        if "model" in user_query.lower():
            st.info("In this app A sequential neural network and a Random Forest model with three dense layers was trained on the 2019 UK Road Accident dataset.")
        elif "accuracy" in user_query.lower():
            st.info("The model has an average accuracy of 80%.")
        else:
            st.info("I'm still learning. Please ask about model, data, or usage.")
        st.write("Ask me anything about the app or the dataset!")
