# from flask import Flask, request, jsonify, render_template
# from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.chat_models import ChatOllama
# from dotenv import load_dotenv
# from src.prompt import system_prompt
# import os

# app = Flask(__name__)

# load_dotenv()

# PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# embeddings = download_hugging_face_embeddings()

# index_name = "medicalbot"

# docsearch = PineconeVectorStore.from_existing_index(
#     index_name=index_name,
#     embedding=embeddings,
# )

# retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

# llm = ChatOllama(
#     model="mistral",  # You can change this to mistral, gemma, etc.
#     # model="llama3",
#     temperature=0.4,
#     max_tokens=500,
#     base_url="http://localhost:11434"
# )

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ]
# )

# question_answer_chain = create_stuff_documents_chain(llm,prompt)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/get", methods=["POST"])
# def chat():
#     msg = request.get_json()["msg"]  # instead of request.form["msg"]
#     print("User Input:", msg)

#     response = rag_chain.invoke({"input": msg})
#     print("Bot Response:", response["answer"])
#     return jsonify({"reply": response["answer"]})  # Wrap it properly


# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0",port=8080)
from flask import Flask, request, jsonify, render_template
from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from dotenv import load_dotenv
from src.prompt import system_prompt
import os

# Initialize Flask
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Pinecone API Key
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY  # Redundant if it's loaded already, but OK

# Load Embeddings
embeddings = download_hugging_face_embeddings()

# Load VectorStore (assuming index already created)
index_name = "medicalbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)

# Create Retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Set up Ollama LLM
llm = ChatOllama(
    model="llama3.2",  # Or "llama3", "gemma", etc.
    temperature=0.4,
    max_tokens=500,
    base_url="http://localhost:11434"
)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Create QA chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])  # Match this with your JS fetch() endpoint
def chat():
    msg = request.get_json().get("message")  # Match the key sent from JS
    if not msg:
        return jsonify({"reply": "No message received."})

    print("User Input:", msg)
    response = rag_chain.invoke({"input": msg})
    print("Bot Response:", response["answer"])
    return jsonify({"reply": response["answer"]})

# Main entry
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
