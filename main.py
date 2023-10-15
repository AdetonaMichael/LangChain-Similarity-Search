import os
import openai
import pinecone

from dotenv import load_dotenv, dotenv_values
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import pinecone


directory = "docs/business_letters"

def load_docs(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents


documents = load_docs(directory)
print("Documents Count: ", len(documents))


def split_docs(documents, chunck_size=1000, chunck_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunck_size, chunk_overlap=chunck_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

docs = split_docs(documents)
print("Chuncks Count: ", len(docs))
print(docs[0].page_content)


