import os
from dotenv import load_dotenv
import streamlit as st

def load_key():
    dotenv_path = "key.env"
    load_dotenv(dotenv_path)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(f"no key")
    return api_key


def main():
    st.title("PDF Summarizer")
    st.divider()

    try:
        os.environ["GOOGLE_API_KEY"] = load_key()
    except ValueError as err:
        st.error(str(err))
        return

    pdf = st.file_uploader('Upload your PDF Document', type='pdf')

    st.subheader('pdf:')
    st.write(pdf)

if __name__ == '__main__':
    main()