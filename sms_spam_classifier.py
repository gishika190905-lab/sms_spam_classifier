
import streamlit as st
import pickle
import nltk
import string
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📩",
    layout="centered"
)

# ---------------- SESSION STATE ----------------
if "show_result" not in st.session_state:
    st.session_state["show_result"] = False

if "sms_input" not in st.session_state:
    st.session_state["sms_input"] = ""

# ---------------- NLTK RESOURCES ----------------
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# ---------------- TEXT PREPROCESSING ----------------
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))
punctuations = string.punctuation


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stop_words and i not in punctuations:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# ---------------- LOAD SAVED MODEL ----------------
BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "model.pkl", "rb") as f:
    model = pickle.load(f)

with open(BASE_DIR / "vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


# ---------------- CALLBACK FUNCTIONS ----------------
def clear_result():
    st.session_state["show_result"] = False


def load_sample(message):
    st.session_state["sms_input"] = message
    st.session_state["show_result"] = False


def reset_app():
    st.session_state["sms_input"] = ""
    st.session_state["show_result"] = False
    st.session_state.pop("prediction", None)


# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #F5F9FF 0%, #FFFFFF 65%);
    color: #24324A;
}

.block-container {
    max-width: 760px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1 {
    color: #244B80;
    text-align: center;
    font-size: 2.35rem !important;
    font-weight: 750 !important;
}

h2, h3 {
    color: #244B80;
}

.subtitle {
    text-align: center;
    color: #718096;
    font-size: 1.05rem;
    margin-top: -8px;
    margin-bottom: 22px;
}

div[data-testid="stTextArea"] textarea {
    background-color: #FFFFFF;
    color: #24324A;
    border: 1px solid #D6E5F7;
    border-radius: 14px;
    font-size: 16px;
    padding: 14px;
}

div[data-testid="stTextArea"] textarea:focus {
    border-color: #78A9EC;
    box-shadow: 0 0 0 2px #E5F0FF;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    border: 1px solid #D6E5F7;
    background: #FFFFFF;
    color: #315A91;
    font-weight: 600;
    min-height: 43px;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border-color: #78A9EC;
    background: #EDF5FF;
    color: #244B80;
}

div.stButton > button[kind="primary"] {
    background: #3978D4;
    color: #FFFFFF;
    border: none;
}

div.stButton > button[kind="primary"]:hover {
    background: #285FAF;
    color: #FFFFFF;
}

.result-card {
    padding: 22px;
    border-radius: 16px;
    margin: 18px 0;
    text-align: center;
    border: 1px solid;
    box-shadow: 0 5px 18px rgba(36, 75, 128, 0.07);
}

.result-card.spam {
    background: #FFF1F2;
    border-color: #FECDD3;
    color: #9F1239;
}

.result-card.ham {
    background: #ECFDF5;
    border-color: #A7F3D0;
    color: #166534;
}

.result-icon {
    font-size: 35px;
    margin-bottom: 5px;
}

.result-title {
    font-size: 23px;
    font-weight: 750;
    margin-bottom: 7px;
}

.result-description {
    font-size: 14px;
    line-height: 1.6;
}

div[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #E0EAF7;
    border-radius: 12px;
}

.footer {
    color: #8290A5;
    text-align: center;
    font-size: 12px;
    margin-top: 24px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- PAGE HEADER ----------------
st.markdown(
    "<h1>📩 SMS Spam Classifier</h1>",
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">'
    'An intelligent way to check suspicious text messages.'
    '</p>',
    unsafe_allow_html=True
)

st.divider()

# ---------------- MESSAGE INPUT ----------------
st.markdown("### 💬 Enter your message")

st.caption(
    "Paste an SMS below and let the machine-learning model classify it."
)

input_sms = st.text_area(
    "SMS message",
    placeholder="Example: Congratulations! You have won a free prize...",
    height=145,
    key="sms_input",
    on_change=clear_result,
    label_visibility="collapsed"
)

st.caption(f"{len(input_sms)} characters")

# ---------------- SAMPLE MESSAGES ----------------
st.markdown("**Try a sample message**")

sample_col1, sample_col2 = st.columns(2)

with sample_col1:
    st.button(
        "🚨 Load spam example",
        on_click=load_sample,
        args=(
            "Congratulations! You have won a free prize. "
            "Call now to claim your reward!",
        )
    )

with sample_col2:
    st.button(
        "💬 Load normal example",
        on_click=load_sample,
        args=("Hey, are we meeting at the library today?",)
    )

# ---------------- ACTION BUTTONS ----------------
check_col, clear_col = st.columns([3, 1])

with check_col:
    check_clicked = st.button(
        "🔍 Check Message",
        type="primary"
    )

with clear_col:
    st.button(
        "↻ Clear",
        on_click=reset_app
    )

# ---------------- PREDICTION ----------------
if check_clicked:
    if not input_sms.strip():
        st.session_state["show_result"] = False
        st.warning("Please enter a message before checking.")

    else:
        try:
            transformed_sms = transform_text(input_sms)
            vector_input = vectorizer.transform([transformed_sms])
            prediction = model.predict(vector_input)[0]

            st.session_state["prediction"] = prediction
            st.session_state["show_result"] = True

        except Exception as e:
            st.session_state["show_result"] = False
            st.error(f"Unable to classify this message: {e}")

# ---------------- RESULT CARD ----------------
if st.session_state["show_result"]:
    prediction = st.session_state["prediction"]

    if prediction == 1:
        st.markdown("""
        <div class="result-card spam">
            <div class="result-icon">🚨</div>
            <div class="result-title">SPAM DETECTED</div>
            <div class="result-description">
                This message has been classified as spam.
                Avoid clicking suspicious links or sharing personal
                information with untrusted senders.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="result-card ham">
            <div class="result-icon">✅</div>
            <div class="result-title">NOT SPAM (HAM)</div>
            <div class="result-description">
                This message has been classified as not spam.
                Always use caution with unexpected links and requests.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.caption(
        "This is a model prediction, not a guarantee that a message "
        "is safe or malicious."
    )

# ---------------- ABOUT THE MODEL ----------------
st.divider()

with st.expander("ℹ️ About this project"):
    st.markdown("""
    ### How does it work?

    This application uses machine learning to classify SMS messages
    as **spam** or **not spam (ham)**.

    **Technologies used**
    - Python — application logic
    - Streamlit — interactive web interface
    - NLTK — tokenization, stopword removal and stemming
    - TF-IDF — converts processed text into numerical features
    - Multinomial Naive Bayes — predicts the message category

    **Prediction workflow**

    SMS input → Text preprocessing → TF-IDF vectorization
    → Trained model → Classification result

    The app uses a previously trained model and vectorizer.
    It does not retrain the model when you submit a message.
    """)