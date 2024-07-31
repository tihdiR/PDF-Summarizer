import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Import css file.
with open('style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Loads api key from key.env file.
def load_api_key():
    dotenv_path = "key.env"
    load_dotenv(dotenv_path)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(f"Unable to retrieve GOOGLE_API_KEY from {dotenv_path}")
    return api_key

# Splits text into chunks and uses Google embeddings to vectorize the text.
# Similiarity search is performed on the chunks and embeddings to produce a knowledge base.
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

    # Initialize session states.
    if 'pdf_text' not in st.session_state:
        st.session_state['pdf_text'] = None
        st.session_state['knowledgeBase'] = None

    # Checks if no document has been uploaded yet.
    if st.session_state['pdf_text'] is None:
        st.markdown('<div class="file-uploader">', unsafe_allow_html=True)
        pdf = st.file_uploader('Upload your PDF Document', type='pdf')
        st.markdown('</div>', unsafe_allow_html=True)

        if pdf:
            pdf_reader = PdfReader(pdf)
            text = ""

            st.markdown('<span id="input"></span>', unsafe_allow_html=True)
            st.markdown('<p style="text-align:center;">Enter the range of pages you wish to analyze (default is all of the pages).</p>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                page_start = st.number_input('Page start (1-indexed)', min_value=1, max_value=len(pdf_reader.pages), value=1)
            with col2:
                page_end = st.number_input('Page end (1-indexed)', min_value=page_start, max_value=len(pdf_reader.pages), value=len(pdf_reader.pages))

            st.markdown('<span id="button-after"></span>', unsafe_allow_html=True)
            doc_clicked = st.button("Process document")

            # Processes document using the range of pages inputted.
            if doc_clicked:
                with st.spinner("Processing..."):
                    for page_num in range(page_start - 1, page_end):
                        page = pdf_reader.pages[page_num]
                        content = page.extract_text()
                        if content:
                            text += content

                    st.session_state['pdf_text'] = text
                    st.session_state['knowledgeBase'] = process_text(text)
                    st.markdown('<p style="text-align:center;">Document Processed!</p>', unsafe_allow_html=True)

    if st.session_state['pdf_text'] is not None:
        text = st.session_state['pdf_text']
        knowledgeBase = st.session_state['knowledgeBase']

        summary_length = st.selectbox('Select Summary Length', ['Brief Summary (1 paragraph)', 'Regular Summary (2-4 paragraphs)', 'Detailed Summary (1 page)'])

        st.markdown('<span id="button-after"></span>', unsafe_allow_html=True)
        gen_summary_clicked = st.button("Generate Summary!")

        # Generates summary based on summary length provided and uses the respective prompt.
        if gen_summary_clicked:
            with st.spinner("Generating summary..."):
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

        st.markdown('<h2>Ask a Question:</h2>', unsafe_allow_html=True)
        user_query = st.text_input('Enter your question about the document:')
        ask_query_clicked = st.button("Ask Question")

        # Uses a user query to ask about the document
        if ask_query_clicked and user_query:
            with st.spinner("Searching for answer..."):
                user_query = "Answer the question: " + user_query + "using the information provided in the document. Give a detailed and comprehensive answer."
                docs = knowledgeBase.similarity_search(user_query)
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")
                chain = load_qa_chain(llm, chain_type='stuff')
                response = chain.run(input_documents=docs, question=user_query)
                st.markdown('<h2>Answer:</h2>', unsafe_allow_html=True)
                st.markdown(response, unsafe_allow_html=True)

        # Option to upload a different document which resets the sessions states and reruns the program.
        if st.button("Upload a Different Document"):
            st.session_state['pdf_text'] = None
            st.session_state['knowledgeBase'] = None
            st.rerun()
        
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
