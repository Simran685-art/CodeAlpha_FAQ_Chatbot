import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

st.set_page_config(page_title="FAQ Chatbot", page_icon="🤖", layout="wide")

# Session State
if "history" not in st.session_state:
    st.session_state.history = []

st.title("🤖 FAQ Chatbot")
st.write("Ask questions and get answers instantly.")

# Sidebar
st.sidebar.title("📚 FAQ Categories")

categories = [
    "Artificial Intelligence",
    "Machine Learning",
    "Deep Learning",
    "Natural Language Processing",
    "Python",
    "Data Science",
    "Computer Vision",
    "Libraries & Frameworks"
]

for cat in categories:
    st.sidebar.write(f"✅ {cat}")

st.sidebar.markdown("---")

if st.sidebar.button("🗑 Clear History"):
    st.session_state.history = []

st.sidebar.subheader("📜 Chat History")

if st.session_state.history:
    for item in reversed(st.session_state.history):
        st.sidebar.write(item)
else:
    st.sidebar.write("No chats yet.")

# FAQ Database
faq_data = {

    "What is AI?":
    "AI stands for Artificial Intelligence.",

    "AI full form":
    "AI stands for Artificial Intelligence.",

    "Define AI":
    "Artificial Intelligence is the simulation of human intelligence by machines.",

    "What is ML?":
    "ML stands for Machine Learning.",

    "ML full form":
    "ML stands for Machine Learning.",

    "Define Machine Learning":
    "Machine Learning is a subset of AI that learns from data.",

    "What is DL?":
    "DL stands for Deep Learning.",

    "DL full form":
    "DL stands for Deep Learning.",

    "What is NLP?":
    "NLP stands for Natural Language Processing.",

    "NLP full form":
    "NLP stands for Natural Language Processing.",

    "What is Python?":
    "Python is a high-level programming language.",

    "What is Data Science?":
    "Data Science is the process of extracting insights from data.",

    "What is Computer Vision?":
    "Computer Vision enables computers to understand images and videos.",

    "What is Streamlit?":
    "Streamlit is a Python framework used to build web applications.",

    "What is a Chatbot?":
    "A chatbot is software that simulates human conversation.",

    "What is Neural Network?":
    "A Neural Network is inspired by the structure of the human brain.",

    "What is Generative AI?":
    "Generative AI creates new content such as text, images and code.",

    "What is ChatGPT?":
    "ChatGPT is an AI chatbot developed by OpenAI.",

    "What is Supervised Learning?":
    "Supervised Learning uses labeled data for training.",

    "What is Unsupervised Learning?":
    "Unsupervised Learning finds patterns in unlabeled data.",

    "What is Reinforcement Learning?":
    "Reinforcement Learning learns using rewards and penalties.",

    "What is Big Data?":
    "Big Data refers to extremely large datasets.",

    "What is Data Mining?":
    "Data Mining is the process of discovering patterns in data.",

    "What is CNN?":
    "CNN stands for Convolutional Neural Network.",

    "CNN full form":
    "CNN stands for Convolutional Neural Network.",

    "What is RNN?":
    "RNN stands for Recurrent Neural Network.",

    "RNN full form":
    "RNN stands for Recurrent Neural Network.",

    "What is TensorFlow?":
    "TensorFlow is an open-source machine learning framework.",

    "What is Pandas?":
    "Pandas is a Python library used for data analysis.",

    "What is NumPy?":
    "NumPy is a Python library for numerical computing.",

    "What is Scikit Learn?":
    "Scikit-Learn is a machine learning library for Python.",

    "What is OpenCV?":
    "OpenCV is an open-source computer vision library.",

    "What is CodeAlpha?":
    "CodeAlpha provides internships and project-based learning opportunities."
}

questions = list(faq_data.keys())

st.info(f"📊 Total FAQs Available: {len(questions)}")

user_query = st.text_input("💬 Ask Your Question")

if st.button("🚀 Get Answer"):

    if user_query.strip():

        corpus = questions + [user_query]

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(corpus)

        similarity = cosine_similarity(
            vectors[-1],
            vectors[:-1]
        )

        best_score = similarity.max()
        confidence = round(best_score * 100, 2)

        if best_score < 0.25:
            st.error("❌ Sorry, I don't have information about that topic.")
        else:

            best_match = similarity.argmax()
            answer = faq_data[questions[best_match]]

            st.success("✅ Answer Found")

            st.subheader("Answer")
            st.write(answer)

            st.write(f"🎯 Confidence Score: {confidence}%")

            current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

            st.caption(f"🕒 {current_time}")

            st.session_state.history.append(
                f"Q: {user_query}"
            )

    else:
        st.warning("⚠ Please enter a question.")