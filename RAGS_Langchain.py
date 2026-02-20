# Import necessary libraries
import openai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv() # Loads the .env file


llm = OpenAI() # Key is loaded from .env and generated from openAI
llm.invoke("What is EDA?")

pdf_reader = PyPDFLoader("RAGPaper.pdf")
documents = pdf_reader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, ) # used for splitting text into chunks
chunks = text_splitter.split_documents(documents)


# Create embeddings
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents=chunks, embedding=embeddings) #FAISS is used for similarity search and clustering of dense vectors
# documents and embeddings is stored in db

# Adapt if needed
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:""")

qa = ConversationalRetrievalChain.from_llm(llm=llm,
                                           retriever=db.as_retriever(),
                                           condense_question_prompt=CONDENSE_QUESTION_PROMPT,
                                           return_source_documents=True,
                                           verbose=False)

chat_history = []
query = """Does the vendor have experience with similar industries and use cases?​?​​"""
result = qa({"question": query, "chat_history": chat_history})
print(result["answer"])

chat_history = []
query = """?Does the vendor's financial offer make sense compared to the timeline, resources, and deliverables??"""
result = qa({"question": query, "chat_history": chat_history})
print(result["answer"])

chat_history = []
query = """?What is RAGs?"""
result = qa({"question": query, "chat_history": chat_history})
print(result["answer"])