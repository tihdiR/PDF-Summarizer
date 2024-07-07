import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

with open('style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

def load_api_key():
    dotenv_path = "key.env"
    load_dotenv(dotenv_path)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(f"Unable to retrieve GOOGLE_API_KEY from {dotenv_path}")
    return api_key

def process_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=50000,
        chunk_overlap=7000,
    )

    chunks = text_splitter.split_text(text)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    knowledgeBase = FAISS.from_texts(chunks, embeddings)

    return knowledgeBase

def main():
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<h1 class="title">PDF Summarizer</h1>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    try:
        os.environ["GOOGLE_API_KEY"] = load_api_key()
    except ValueError as err:
        st.error(str(err))
        return

    st.markdown('<div class="file-uploader">', unsafe_allow_html=True)
    pdf = st.file_uploader('Upload your PDF Document', type='pdf')
    st.markdown('</div>', unsafe_allow_html=True)

    if pdf:
        pdf_reader = PdfReader(pdf)

        text = ""

        st.markdown('<span id="input"></span>', unsafe_allow_html=True)

        page_start = st.number_input('Page start (1-indexed)', min_value=1, max_value=len(pdf_reader.pages), value=1)
        page_end = st.number_input('Page end (1-indexed)', min_value=page_start, max_value=len(pdf_reader.pages), value=len(pdf_reader.pages))
        summary_length = st.selectbox('Select Summary Length', ['Brief Summary (1 paragraph)', 'Regular Summary (2-4 paragraphs)', 'Detailed Summary (1 page)'])

        st.markdown('<span id="button-after"></span>', unsafe_allow_html=True)
        gen_summary_clicked = st.button("Generate Summary!")
        if gen_summary_clicked:
            with st.spinner("Generating summary..."):
                for page_num in range(page_start - 1, page_end):  # Adjust for 0-indexed pages
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()

                knowledgeBase = process_text(text)

                query_brief = """Give a brief summary of the content of the document in 3 to 5 sentences.
                Try to capture the main ideas and key points of the document. """

                query_summary = """Give a summary of the content of the document.
                Try to capture the main ideas and key points of the document."""

                query_detailed = """Give a very detailed summary of the content of the document. Write as much as possible.
                Try to capture the main ideas and key points of the document."""

                if summary_length == 'Brief Summary (1 paragraph)':
                    query = query_brief
                elif summary_length == 'Regular Summary (2-4 paragraphs)':
                    query = query_summary
                else:
                    query = query_detailed

                docs = knowledgeBase.similarity_search(query)
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")
                chain = load_qa_chain(llm, chain_type='stuff')
                response = chain.run(input_documents=docs, question=query)

                st.markdown('<h2>Summary Generated:</h2>', unsafe_allow_html=True)
                st.markdown(response, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
