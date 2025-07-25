import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain

# 🔐 Google API Key (you can also use st.secrets later)
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
 # Replace with your real key

# 📚 Load and cache vector store
@st.cache_resource(show_spinner="🔄 Indexing medical PDF, please wait...")
def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if os.path.exists("faiss_index/index.faiss"):
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    else:
        loader = PyPDFLoader("the gale encyclopedia of medicine.pdf")  # PDF should be in same folder
        pages = loader.load_and_split()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(pages)
        vector = FAISS.from_documents(docs, embeddings)
        vector.save_local("faiss_index")
        return vector

vector_store = load_vector_store()
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# 💬 Intent Detection
def detect_intent(question: str) -> str:
    q = question.lower().strip()
    if any(x in q for x in ["hi", "hello", "hey", "who are you"]):
        return "greeting"
    if any(x in q for x in ["thank you", "thanks", "thx"]):
        return "gratitude"
    if len(q.split()) <= 2:
        return "about"
    return "medical"

# 🧠 Custom Prompt
custom_prompt = PromptTemplate.from_template('''
You are **PharmaAI**, a highly professional AI assistant trained to provide accurate, safe, and medically relevant information from reliable pharmaceutical references.

Respond based on intent:

1. **Greeting** → "Hello! I'm PharmaAI, your assistant for pharmaceutical and medical queries. Feel free to ask anything."
2. **Gratitude** → "You're welcome! Let me know if there's anything else I can help with."
3. **About** → "I'm PharmaAI, designed to provide trustworthy medical insights using reliable sources. Ask me any medicine or health-related question."
4. **Medical** →
   - Use one of these openers:
     - "📘 According to the available medical information:"
     - "💊 Based on reliable pharmaceutical references:"
     - "🧠 PharmaAI suggests the following information for your inquiry:"
     - "🩺 Here's what I found regarding your query:"
   - If context is missing, reply:
     "I'm sorry, I couldn't find that information at the moment. Let me know if there's anything else I can assist you with."
   - Optionally, append these safe disclaimers as relevant:
     - "⚠️ This is general medical information and should not replace advice from a licensed healthcare provider."
     - "📌 Always consult a qualified doctor before starting or stopping any medication."
     - "⚕️ Proper diagnosis is essential before using any treatment for symptoms."

---
Context:
{context}
---
User Question:
{question}
Answer:
''')

# 🧠 LLM and QA Chain
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
qa_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=custom_prompt)
qa_with_retriever = RetrievalQA(combine_documents_chain=qa_chain, retriever=retriever)

# 🌐 Streamlit UI
st.set_page_config(page_title="Pharma AI", layout="wide")
st.title("💊 PharmaAI")
st.markdown("Ask me about any **medicine**, **disease**, **side effects**, or **usage**. Powered by Gemini & LangChain.")

# 💾 Chat History
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# 🧹 Clear Chat Button
if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []

# 🔍 User Query
query = st.text_input("Ask your question:")

if query:
    with st.spinner("🤔 Thinking..."):
        intent = detect_intent(query)

        if intent in ["greeting", "gratitude", "about"]:
            responses = {
                "greeting": "👋 Hello! I'm PharmaAI, your assistant for pharmaceutical and medical queries. Feel free to ask anything.",
                "gratitude": "🙏 You're welcome! Let me know if there's anything else I can help with.",
                "about": "ℹ️ I'm PharmaAI, designed to provide trustworthy medical insights using reliable sources. Ask me any medicine or health-related question."
            }
            answer = responses[intent]
        else:
            answer = qa_with_retriever.run(query)

        # Add to chat history
        st.session_state.chat_history.append(("🧑 You", query))
        st.session_state.chat_history.append(("🤖 PharmaAI", answer))

# 🗨️ Show chat history
for sender, msg in st.session_state.chat_history:
    with st.chat_message(sender.split()[1].lower()):
        st.markdown(f"**{sender}:**\n\n{msg}")
