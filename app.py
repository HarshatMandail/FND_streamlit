import streamlit as st
import joblib

# Load pipeline (model already includes vectorizer + classifier)
model = joblib.load("fake_news_model.pkl")

# Title
st.title("📰 Fake News Detector")
st.subheader("Enter a news article to predict whether it's Real or Fake")

# Input
news_text = st.text_area("Enter News Content:")

if st.button("Predict"):
    if news_text.strip() == "":
        st.warning("Please enter some text!")
    else:
        try:
            # ✅ Directly predict with pipeline
            prediction = model.predict([news_text])[0]

            # Map output
            label_map = {0: "Fake", 1: "Real"}
            output = label_map.get(prediction, prediction)

            st.success(f"This news is **{output}**.")
        except Exception as e:
            st.error(f"An error occurred during prediction:\n\n{e}")

st.markdown("---")
st.markdown("Made with ❤️ using Machine Learning & Streamlit")