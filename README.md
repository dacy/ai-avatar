# AI Avatar - Voice Search Tool

A semantic search tool that processes text documents and answers questions about their content. The tool uses semantic embeddings and vector similarity search to find relevant information, and a question-answering model to extract precise answers.

## Features

- **Text-based Search**: Query your knowledge base using natural language
- **Semantic Search**: Uses FAISS for efficient vector similarity search
- **Question Answering**: Uses Ollama for generating answers from context
- **On-Premise Processing**: All data processing happens locally for privacy
- **Modern UI**: Clean, responsive interface with real-time feedback
- Support for both CPU and GPU processing
- Real-time system status monitoring
- Markdown rendering for formatted responses

## System Requirements

- Python 3.8+
- Ollama installed and running locally
- FAISS library for vector similarity search

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-avatar.git
cd ai-avatar
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the server:
```bash
python full_server.py
```

2. Open your browser and navigate to `http://localhost:5000`

3. Search your knowledge base:
   - Type your question in the text input
   - Click "Submit Question" or press Enter
   - The system will search through indexed documents and provide an answer

## System Status

- **Frontend**: HTML, CSS, JavaScript with modern UI components
- **Backend**: Python Flask server
- **Search Engine**: FAISS for vector similarity search
- **LLM**: API interface with Ollama or function invokatino using Transformers for question answering
- **Vector Database**: FAISS for efficient similarity search
- **Embedding Model**: SentenceTransformer with "all-MiniLM-L6-v2"

## Project Structure

### Implemented Features
- ✅ Text-based search interface
- ✅ Semantic search with FAISS
- ✅ Question answering with Ollama
- ✅ Modern, responsive UI
- ✅ Real-time status updates
- ✅ Error handling and logging

### In Progress
- 🔄 Voice input interface
- 🔄 Voice output
- 🔄 Confluence integration

## Development

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Ollama for providing the LLM capabilities
- FAISS for efficient similarity search
- SentenceTransformer for text embeddings 