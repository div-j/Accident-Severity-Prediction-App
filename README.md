
# 🚨 Road Accident Severity Prediction App

This interactive Streamlit app predicts the **severity of road accidents** using a machine learning model trained on the **UK 2019 Road Accident Dataset**. It also features an AI-powered chat assistant powered by **Gemini** to help users understand the model, data, and usage.

## 🎯 Project Objective

To assist UK emergency services by predicting the **severity level** of a road accident — `Fatal`, `Serious`, or `Slight` — using environmental and driver-related features. The app uses a trained **Keras neural network model**, compares it with traditional ML models, and allows real-time prediction and explanations.

---

## 📦 Features

- ✅ Predict accident severity using form inputs
- 🧠 Uses a trained **Keras model and Random Forest Model** for inference
- 💬 AI Chat Assistant powered by **Gemini** to answer user queries
- 🧮 Input preprocessing using **LabelEncoders** and **Scalers**
- 📈 Clean, intuitive Streamlit UI

---

## ⚙️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Keras (TensorFlow)
- **AI Chat**: Custom (Comming soon -> Gemini (Google Generative AI))
- **Language**: Python 3.9+
- **Libraries**: `streamlit`, `tensorflow`, `scikit-learn`,`pandas`, `numpy`, `google.generativeai`

---

## 🛠️ How to Run Locally

1. **Clone the repository**
```bash
git clone https://github.com/div-j/Accident-Severity-Prediction-App.git

cd accident-severity-streamlit
````

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Add your Gemini API key**

Create a `.env` file and add:

```bash
GOOGLE_API_KEY=your_gemini_api_key
```

4. **Ensure model files are available**

Place your trained model and preprocessing files:

* `accident_severity_model.h5`
* `label_encoders.pkl`
* `scaler.pkl`

5. **Run the app**

```bash
streamlit run app.py
```

---

## 📊 Dataset

Dataset: UK Road Accident Data (2019)
Features used include:

* Speed limit
* Light & weather conditions
* Road surface
* Vehicle type
* Driver age and sex
* Junction location
* Accident impact point
* And more...

---

## 🤖 Model Info

* **Model**: Neural Network (Keras)
* **Classes**: `Fatal`, `Serious`, `Slight`
* **Accuracy**: \~80% on test data
* Preprocessing: StandardScaler + LabelEncoders

---

## 💬 Gemini Chat Assistant

Ask the assistant anything like:

* “What model is used?”
* “How accurate is it?”
* “What features are important?”
* “How is severity predicted?”

---

## ✨ Sample Use Case

> "A user inputs accident conditions such as rainy weather, slippery road, vehicle type as motorcycle, and the app predicts a high chance of **‘Serious’** accident with 87% confidence."

---



## 👨‍💻 Author

Developed by **John Ibe Divine**

Affiliated with Engi-Tech
“**We Train, We Build, and We Deploy**”

---

## 📜 License

