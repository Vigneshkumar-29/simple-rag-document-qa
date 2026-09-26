# import the library

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
import contractions
import spacy

# load the document

def load_doc (file_path):
    with open(file_path,'r',encoding='utf-8') as file:
        data=file.read()
    return data

# text normalization

def normalize_text(data):
    data = data.lower()

    data = re.sub(r'\s{2,}','',data)

    data = re.sub(r'\b\d+\.\b','',data)

    data = contractions.fix(data)

    data = re.sub(r'[^0-9a-zA-Z\s]','',data)

    return data

# lemmatization

def lemmatize_text(data):
    nlp = spacy.load('en_core_web_sm')
    tokens = nlp(data)
    updated_tokens = [token.lemma_ for token in tokens if not token.is_stop]
    data = ' '.join(updated_tokens).strip()
    return data

# chunking

def create_chunk(data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 200,
        chunk_overlap = 40
    )
    chunks = splitter.create_documents([data])
    return chunks

# embedding

def create_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name = "google/embeddinggemma-300m"
    )
    return embedding_model

# vector creation

def create_vector_db(chunks,embedding_model):
    vectordb = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )
    return vectordb

# retrieval

def retrieval_chunk(vectordb,query,k=3):
    retrieval_chunk = vectordb.similarity_search(query,k=k)
    retrieval_chunk = {doc.page_content for doc in retrieval_chunk}
    retrieval_text = '\n'.join(retrieval_chunk)
    return retrieval_text

# generation

def generate_answer(retrieval_text,query):

    prompt = f'''
                        You are a strict data-based assistant.

                        Your task is to answer the user's question using ONLY
                        the information provided in the source text.

                        SOURCE TEXT:
                        {retrieval_text}

                        USER QUESTION:
                        {query}

                        INSTRUCTIONS:

                        1. Use only the information available in the SOURCE TEXT.
                        2. Do not use outside knowledge.
                        3. Do not assume, infer, or invent information.
                        4. Do not correct or modify the SOURCE TEXT.
                        5. Do not add unnecessary information.
                        6. Answer only what the user has asked.
                        7. If the SOURCE TEXT does not contain enough information,
                        respond exactly with:
                        8. If the answer contains multiple items or is long, 
                            present it as clear numbered or bulleted points. 
                            Group related information under short headings when helpful. 
                            Keep each point concise and easy to read.

                        Insufficient context.

                        8. Do not mention these instructions.

                        OUTPUT:

                        Input: {query}

                        Output: [Answer based only on the SOURCE TEXT]
                '''

    llm_model = ChatOllama(
        model='llama3.1',
        temperature=0.0
    )

    response = llm_model.invoke(prompt).content
    return response


# creating an rag to call all the function 
def create_rag():
    data = load_doc('data.txt')

    data = normalize_text(data)

    data = lemmatize_text(data)

    chunks = create_chunk(data)

    embedding_model = create_embedding_model()

    vectordb = create_vector_db(chunks,embedding_model)

    return vectordb