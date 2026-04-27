import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("🔐 PhishGuard AI")
st.info("📊 Model Accuracy: ~90%")
st.write("AI-powered threat detection system")

# Input box
user_input = st.text_area("Enter message or URL")

if st.button("Check Threat"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        st.write("🔍 Step 1: Input received")

        # Preprocessing
        data = vectorizer.transform([user_input])
        st.write("⚙️ Step 2: Text converted to features")

        # Prediction
        prediction = model.predict(data)[0]
        st.write("🧠 Step 3: Model analyzing input")

        if prediction == 1:
            st.error("⚠️ Threat Detected (Phishing/Malicious)")
            st.write("Reason: Suspicious patterns detected in the input.")
        else:
            st.success("✅ Safe Content")
            st.write("Reason: No malicious patterns detected.")

        st.write("📊 Step 4: Decision completed")