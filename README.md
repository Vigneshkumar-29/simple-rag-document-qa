# Simple RAG Question Answering

A small Python retrieval-augmented generation (RAG) application that answers questions using the content of `data.txt` through a Gradio web interface.

The application preprocesses the document, splits it into chunks, creates vector embeddings, retrieves the three most relevant chunks for each question, and sends only that retrieved context to a local Ollama language model.

## User Interface

The application uses [Gradio](https://www.gradio.app/) to provide the web interface for entering questions and displaying answers.

![Gradio document question answering interface](assets/gradio_ui.png)

## Technology

- **Gradio** - web interface
- **FAISS** - vector similarity search
- **spaCy** - text lemmatization
- **Hugging Face Embeddings** - document and query embeddings
- **Ollama** - local language model inference

## How It Works

1. Loads `data.txt` when the application starts.
2. Normalizes the text and removes stop words through spaCy lemmatization.
3. Splits the document into overlapping chunks.
4. Creates embeddings with `google/embeddinggemma-300m`.
5. Stores the embeddings in a FAISS vector database in memory.
6. Retrieves the three most relevant chunks for each query.
7. Generates an answer with Ollama using the `llama3.1` model.
8. Displays the answer in the Gradio interface.

## Requirements

- Python 3.10 or newer
- The Python packages listed in `requirements.txt`
- [Ollama](https://ollama.com/)
- The Ollama `llama3.1` model
- Internet access on the first run to download Python and Hugging Face model dependencies

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Install and start Ollama, then download the language model:

```bash
ollama pull llama3.1
```

## Run

Make sure `data.txt` is in the project directory and that Ollama is running, then run:

```bash
python main.py
```

Gradio starts a local web interface and prints a URL in the terminal. Open that URL in a browser.

Use the following controls:

- **Generate** - retrieves context and generates an answer.
- **Clear** - clears the question, answer, and status fields.
- **Break** - cancels the current answer-generation request.

## Project Files

- `main.py` - Starts the Gradio interface and connects the UI controls to the RAG pipeline.
- `rag.py` - Contains document loading, preprocessing, chunking, embedding, retrieval, and answer generation functions.
- `data.txt` - Source document used as the knowledge base.
- `requirements.txt` - Python dependencies.

## Notes

- Answers are restricted to the retrieved text. When the context is insufficient, the model is instructed to return `Insufficient context.`
- The FAISS vector database is created in memory each time the application starts; it is not persisted between runs.
- Replace the contents of `data.txt` to ask questions about a different document.
- The first startup may take time while spaCy and the Hugging Face embedding model are loaded.
