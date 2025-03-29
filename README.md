# AI Avatar - Voice Search Tool

A semantic search tool that processes text documents and answers questions about their content. The tool uses semantic embeddings and vector similarity search to find relevant information, and a question-answering model to extract precise answers.

## Features

- **Text-based Search**: Query your knowledge base using natural language
- **Semantic Search**: Uses FAISS for efficient vector similarity search
- **Question Answering**: Uses Ollama for generating answers from context
- **On-Premise Processing**: All data processing happens locally for privacy
- **Modern UI**: Clean, responsive interface with real-time feedback
- **Web Crawling**: Add knowledge from web pages with cookie support
- Support for both CPU and GPU processing
- Real-time system status monitoring
- Markdown rendering for formatted responses

## System Requirements

- Python 3.8+
- Ollama installed and running locally
- FAISS library for vector similarity search
- Chrome browser (for web crawling)

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

4. Install Playwright browsers:
```bash
playwright install chromium
```

5. Configure the application:
   - Copy `config/config.yaml.template` to `config/config.yaml`
   - Edit `config/config.yaml` with your settings:
     
     - Configure your Confluence credentials if using Confluence
     - Set up browser profile for web crawling:
       ```yaml
       browser:
         user_data_dir: "folder_you_want_to_store_browser_data"
         headless: false  # Set to false to allow manual login
       ```
     - Adjust model settings as needed
     - Customize other settings like server port, data directories, etc.

## Usage

1. Start the server:
```bash
python full_server.py
```

2. Open your browser and navigate to `http://localhost:8000`

3. Use the web interface to:
   - Add knowledge from web pages or Confluence
   - Search through your knowledge base
   - Use voice commands for hands-free operation

## System Status

- **Frontend**: HTML, CSS, JavaScript with modern UI components
- **Backend**: Python Flask server
- **Search Engine**: FAISS for vector similarity search
- **QA LLM**: API interface with Ollama or function invocation using Transformers for question answering
- **Vector Database**: FAISS for efficient similarity search
- **Embedding Model**: SentenceTransformer with "all-MiniLM-L6-v2"
- **Web Crawler**: Playwright with Chrome profile support

## Project Structure

### Implemented Features
- ✅ Text-based search interface
- ✅ Semantic search with FAISS
- ✅ Question answering with Ollama
- ✅ Modern, responsive UI
- ✅ Real-time status updates
- ✅ Error handling and logging
- ✅ Web crawling with cookie storage

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

- Ollama and Transformers models for providing the LLM QA capabilities
- FAISS for efficient similarity search
- SentenceTransformer for text embeddings 