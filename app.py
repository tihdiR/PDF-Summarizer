import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st
# from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def load_openai_api_key():
    dotenv_path = "key.env"
    load_dotenv(dotenv_path)
    openai_api_key = os.getenv("GOOGLE_API_KEY")
    if not openai_api_key:
        raise ValueError(f"Unable to retrieve GOOGLE_API_KEY from {dotenv_path}")
    return openai_api_key

def process_text(text):
    # Split the text into chunks using Langchain's CharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=50000,
        chunk_overlap=7000,
    )
    chunks = text_splitter.split_text(text)

    # Convert the chunks of text into embeddings to form a knowledge base
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    knowledgeBase = FAISS.from_texts(chunks, embeddings)


    return knowledgeBase

def main():
    st.title("PDF Summarizer")
    st.divider()

    try:
        os.environ["GOOGLE_API_KEY"] = load_openai_api_key()
    except ValueError as e:
        st.error(str(e))
        return

    pdf = st.file_uploader('Upload your PDF Document', type='pdf')

    if pdf:
        pdf_reader = PdfReader(pdf)

        text = ""

        page_start = st.number_input('Page start (1-indexed)', min_value=1, max_value=len(pdf_reader.pages), value=1)
        page_end = st.number_input('Page end (1-indexed)', min_value=page_start, max_value=len(pdf_reader.pages), value=len(pdf_reader.pages))

        if st.button('Generate Summary'):

            for page_num in range(page_start - 1, page_end):  # Adjust for 0-indexed pages
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
                # st.write(f"Processed page {page_num + 1}/{len(pdf_reader.pages)}") 

            # st.write(f"Total text length: {len(text)}")

            knowledgeBase = process_text(text)

            query2 = """Give a brief summary of the content of the document in 3 to 5 sentences.
            Focus on capturing the main ideas and key points discussed in the document. 
            Use your own words and ensure clarity and coherence in the summary."""

            query = """Give a very detalied summary of the content of the document. Write as much as possible.
            Focus on capturing the main ideas and key points discussed in the document. 
            Use your own words and ensure clarity and coherence in the summary."""

            docs = knowledgeBase.similarity_search(query)
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")
            chain = load_qa_chain(llm, chain_type='stuff')
            response = chain.run(input_documents=docs, question=query)

            st.subheader('Summary Results:')
            st.write(response)

if __name__ == '__main__':
    main()