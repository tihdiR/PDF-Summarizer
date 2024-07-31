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
   <img width="877" alt="Screen Shot 2024-07-31 at 4 42 08 PM" src="https://github.com/user-attachments/assets/0db39824-0277-4456-a37a-0189dcb1a4c3">

2. **Select Page Range**: Choose the start and end pages you want to summarize.
   <img width="785" alt="Screen Shot 2024-07-31 at 4 42 58 PM" src="https://github.com/user-attachments/assets/4f46263b-371f-43ec-9468-76f974dad264">

3. **Process Document**: Click on "Process document" to extract and process the text from the selected pages.

4. **Generate Summary**: After processing, choose the desired summary length and click "Generate Summary".
   <img width="745" alt="Screen Shot 2024-07-31 at 4 43 31 PM" src="https://github.com/user-attachments/assets/599c0ae1-9574-499d-bc64-bddef26db2e0">

5. **Ask Questions**: Input a question related to the document's content and click "Ask Question" to get an answer.
   <img width="749" alt="Screen Shot 2024-07-31 at 4 44 30 PM" src="https://github.com/user-attachments/assets/cf1caa56-92dd-4197-b158-618d33fa6499">

6. **Upload a Different Document**: Use the "Upload a Different Document" button to process a new document.

## Dependencies

- **Streamlit**: For building the interactive web application.
- **PyPDF2**: For reading and extracting text from PDF documents.
- **LangChain**: For text processing and LLM integration.
- **Google Generative AI**: For generating embeddings and responses based on the content.

## Contributing

Feel free to fork this repository, create a new branch, and make contributions. Pull requests are welcome.


