import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load("fake_news_model.pkl")

st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detection App")
st.markdown("Enter a news article below to check if it's real or fake.")

news_text = st.text_area("News Article Text", height=200)

if st.button("Check News"):
    if news_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        prediction = model.predict([news_text])[0]
        confidence = model.decision_function([news_text])[0]

        if prediction == 1:
            st.success("This news appears to be **Real** ✅")
        else:
            st.error("This news appears to be **Fake** 🚫")

        

        st.caption(f"Confidence Score: {abs(confidence):.2f}")
st.markdown("---")
st.markdown("Made with ❤️ using Machine Learning & Streamlit")