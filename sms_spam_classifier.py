import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📩",
    layout="centered"
)
# ---------------- TEXT PREPROCESSING ----------------
# Download required NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))
punctuations = string.punctuation
def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stop_words and i not in punctuations:
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)
# ---------------- LOAD TRAINED FILES ----------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background-color: #101827;
    color: #F8FAFC;
}

h1 {
    color: #60A5FA;
    text-align: center;
}

.subtitle {
    text-align: center;
    color: #CBD5E1;
    font-size: 17px;
}

.stTextArea textarea {
    background-color: #1E293B;
    color: white;
    border-radius: 12px;
}

.stButton button {
    background-color: #3B82F6;
    color: white;
    border-radius: 10px;
    border: none;
    width: 100%;
    font-weight: bold;
    padding: 12px;
}

.stButton button:hover {
    background-color: #2563EB;
    color: white;
}
</style>
""", unsafe_allow_html=True)
def clear_result():
    st.session_state["show_result"] = False
# ---------------- FRONTEND ----------------
st.title("📩 SMS Spam Classifier")
st.markdown(
    '<p class="subtitle">Detect spam messages using Machine Learning</p>',
    unsafe_allow_html=True
)
st.divider()
st.write("### Enter your SMS message")
input_sms = st.text_area(
    "SMS Message",
    placeholder="Enter your SMS message here...",
    height=150,
    label_visibility="collapsed",
    key="sms_input",
    on_change=clear_result
)
if "show_result" not in st.session_state:
    st.session_state["show_result"] = False

if st.button("🔍 Check Message"):

    if not input_sms.strip():
        st.session_state["show_result"] = False
        st.warning("Please enter a message first.")

    else:
        transformed_sms = transform_text(input_sms)
        vector_input = vectorizer.transform([transformed_sms])
        prediction = model.predict(vector_input)[0]

        st.session_state["prediction"] = prediction
        st.session_state["show_result"] = True

if st.session_state["show_result"]:

    prediction = st.session_state["prediction"]

    if prediction == 1:
        st.error("🚨 SPAM MESSAGE")
        st.write(
            "This message is classified as spam. "
            "Be careful before clicking links or sharing information."
        )
    else:
        st.success("✅ NOT SPAM ")
        st.write("This message is classified as not spam.")