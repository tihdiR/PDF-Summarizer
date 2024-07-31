# PDF Summarizer Application

This is a PDF Summarizer application built with Streamlit, LangChain, and Google Generative AI. The application allows users to upload a PDF document, process its content, and generate summaries of varying lengths. Additionally, users can ask specific questions about the content of the document.

## Features

- **PDF Upload**: Upload PDF documents for processing.
- **Page Range Selection**: Select specific pages of the PDF to summarize.
- **Summary Generation**: Generate brief, regular, or detailed summaries based on user preference.
- **Question Answering**: Ask specific questions about the content of the document and get accurate responses.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/PDF-Summarizer.git
cd PDF-Summarizer
```
### 2. Install Dependencies

Create a virtual environment and install the required Python packages.

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

### 3. Set Up API Keys

Create a file named **'key.env'** and add your Google API key.
`
GOOGLE_API_KEY=your_google_api_key_here
`

### 4. Run the Application

```
streamlit run app.py
```

## Usage

1. **Upload a PDF**: Click on "Upload your PDF Document" to upload a PDF file.
2. **Select Page Range**: Choose the start and end pages you want to summarize.
3. **Process Document**: Click on "Process document" to extract and process the text from the selected pages.
4. **Generate Summary**: After processing, choose the desired summary length and click "Generate Summary".
5. **Ask Questions**: Input a question related to the document's content and click "Ask Question" to get an answer.
6. **Upload a Different Document**: Use the "Upload a Different Document" button to process a new document.

## Dependencies

- **Streamlit**: For building the interactive web application.
- **PyPDF2**: For reading and extracting text from PDF documents.
- **LangChain**: For text processing and LLM integration.
- **Google Generative AI**: For generating embeddings and responses based on the content.

## Contributing

Feel free to fork this repository, create a new branch, and make contributions. Pull requests are welcome.


