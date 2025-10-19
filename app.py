import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# -------------------- ENV SETUP --------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DB_FAISS_PATH = "vectorstore/db_faiss"

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="🩺 MediBot",
    page_icon="🤖",
    layout="wide",
)

# -------------------- CUSTOM STYLES --------------------
st.markdown("""
    <style>
    /* Transparent background for full app */
    .stApp {
        background: transparent !important;
    }

    /* Remove padding, ensure clean layout */
    .main {
        background: transparent !important;
        color: #0f172a;
    }

    /* Medical-blue subtle gradient background (optional) */
    body {
        background: linear-gradient(to right, #e6f7ff, #ffffff);
        background-attachment: fixed;
    }

    /* Title and subtitles */
    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #475569;
        margin-bottom: 25px;
    }

    /* User and Bot Chat Bubbles */
    .chat-bubble-user {
        background-color: #dbeafe;
        padding: 12px 18px;
        border-radius: 16px;
        margin: 10px 0;
        color: #0f172a;
        font-size: 1rem;
        border-right: 6px solid #2563eb;
    }
    .chat-bubble-bot {
        background-color: #e3f2fd;
        padding: 15px 20px;
        border-radius: 16px;
        margin: 10px 0;
        color: #0f172a;
        font-size: 1rem;
        line-height: 1.6;
        border-left: 6px solid #0284c7;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.05);
    }

    /* Chat titles and footer */
    .chat-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e40af;
        margin-bottom: 5px;
    }
    .footer {
        text-align: center;
        color: #94a3b8;
        margin-top: 30px;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)


# -------------------- HEADER --------------------
st.image("Screenshot 2025-10-19 122118.png", width=160)
# -------------------- HEADER --------------------
st.markdown("""
    <div style="text-align:center; margin-top: -30px;">
        <img src="https://cdn-icons-png.flaticon.com/512/4320/4320337.png" width="80">
        <h1 style="
            font-size: 3rem;
            font-weight: 900;
            color: #0f172a;
            margin-bottom: 0;
            letter-spacing: 1px;
        ">
            MediBot 🤖
        </h1>
        <p style="
            font-size: 1.15rem;
            color: #334155;
            margin-top: 6px;
        ">
            Your AI-powered medical assistant for quick health insights
        </p>
    </div>
""", unsafe_allow_html=True)

st.divider()


# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.image("Screenshot 2025-10-19 122118.png", width=120)
    st.markdown("### 🧬 About MediBot")
    st.write("""
    **MediBot** is an AI-driven medical assistant built using:
    - 🧠 *Groq Llama 3.1 8B Instant* — reasoning engine  
    - 📚 *FAISS Vector Store* — medical knowledge retrieval  
    - 🔗 *LangChain Framework* — RAG pipeline integration  
    - 💬 *Streamlit* — chat interface
    """)
    st.markdown("Developed by **A. Ravi Teja © 2025**")
    st.markdown("---")
    if st.button("🗑️ Reset Chat"):
        st.session_state.messages = []
        st.experimental_rerun()

# -------------------- VECTORSTORE --------------------
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

vectorstore = get_vectorstore()

# -------------------- LLM SETUP --------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.5,
    max_tokens=512,
    api_key=GROQ_API_KEY,
)
retrieval_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
combine_docs_chain = create_stuff_documents_chain(llm, retrieval_prompt)
rag_chain = create_retrieval_chain(vectorstore.as_retriever(search_kwargs={'k': 3}), combine_docs_chain)

# -------------------- MEMORY --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------- DISPLAY CHAT HISTORY --------------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='chat-bubble-user'>🧑‍⚕️ {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bubble-bot'>{msg['content']}</div>", unsafe_allow_html=True)

# -------------------- USER INPUT --------------------
user_prompt = st.chat_input("Ask me a medical question...")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    st.markdown(f"<div class='chat-bubble-user'>🧑‍⚕️ {user_prompt}</div>", unsafe_allow_html=True)

    try:
        enhanced_prompt = f"""
You are a professional **AI medical assistant** providing accurate, evidence-based, and empathetic health information. 
Always use clear, simple, and reassuring language suitable for the general public.

### Structure your response as follows:
1. **Brief Summary** – A short overview of the condition or question in simple terms.
2. **Detailed Explanation** – Explain the topic thoroughly, including:
   - Definition or cause
   - Common symptoms and risk factors
   - How it affects the body
   - Possible treatments or management options
3. **Preventive Measures or Lifestyle Advice** – Include practical, science-backed tips for prevention, 
   such as diet, exercise, vaccinations, and hygiene.
4. **When to Seek Medical Attention** – Indicate warning signs or red flags that require urgent or professional care.
5. **Common Misconceptions (if any)** – Clarify myths or misinformation often associated with the topic.
6. **Medical Disclaimer** – End with this line:
   _"This information is for educational purposes only and is not a substitute for professional medical advice. 
   Always consult a qualified healthcare provider for personalized care."_

Keep your tone: 
- Friendly yet professional 🩺  
- Evidence-based 🧬  
- Non-alarming and supportive 💬  

Use short paragraphs and numbered lists for readability.  

User Query: {user_prompt}
"""


        response = rag_chain.invoke({'input': enhanced_prompt})
        answer = response["answer"]

        answer_html = f"""
        <div class='chat-bubble-bot'>
            <div class='chat-title'>🧠 Medical Insight:</div>
            {answer}
        </div>
        """
        st.markdown(answer_html, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": answer_html})

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")

# -------------------- FOOTER --------------------
st.markdown("<p class='footer'>MediBot 🩺 • Powered by Groq + LangChain + Streamlit</p>", unsafe_allow_html=True)
