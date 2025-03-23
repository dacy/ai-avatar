# AI Avatar - Voice Search Tool

An AI-powered voice search tool that processes Confluence pages and answers questions about their content. The tool uses semantic embeddings and vector similarity search to find relevant information, and a question-answering model to extract precise answers.

## Features

- **Voice Input/Output**: Support for voice queries and text-to-speech responses
- **Confluence Integration**: Process and index Confluence pages for search
- **Semantic Search**: Uses FAISS vector database for efficient similarity search
- **Question Answering**: Extracts precise answers using the RoBERTa QA model
- **On-Premise Processing**: All processing is done locally for data privacy

## Prerequisites

- Python 3.8 or higher
- FFmpeg (for audio processing)
- CUDA-capable GPU (optional, for faster processing)

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

4. Install FFmpeg:
- Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html)
- Linux: `sudo apt-get install ffmpeg`
- Mac: `brew install ffmpeg`

## Configuration

1. Copy the example configuration file:
```bash
cp config/config.yaml.example config/config.yaml
```

2. Edit `config/config.yaml` with your settings:
- Confluence connection details
- Model configurations
- Server settings
- Data storage paths

## Usage

1. Start the server:
```bash
python full_server.py
```

2. Open your browser and navigate to `http://localhost:8000`

3. Process a Confluence page:
   - Enter the Confluence page URL
   - Click "Process Page" to extract and index the content

4. Ask questions:
   - Type your question in the text input
   - Or click the microphone icon to use voice input
   - The system will search the indexed content and provide relevant answers

## Project Structure

```
ai-avatar/
├── config/
│   └── config.yaml          # Configuration settings
├── data/
│   ├── raw/                 # Raw Confluence content
│   ├── processed/           # Processed text chunks
│   └── vectors/             # FAISS index and embeddings
├── src/
│   ├── confluence/          # Confluence crawler
│   ├── embedding/           # Text embedding model
│   ├── qa/                  # Question answering model
│   └── search/              # Vector search implementation
├── static/                  # Static web assets
├── templates/               # HTML templates
├── full_server.py          # Main server application
└── requirements.txt        # Python dependencies
```

## Models Used

- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions)
- **QA Model**: `deepset/roberta-base-squad2`
- **Speech Recognition**: OpenAI Whisper (optional)
- **Text-to-Speech**: Piper TTS (optional)

## Development

### Adding New Features

1. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and commit:
```bash
git add .
git commit -m "Add: your feature description"
```

3. Push changes and create a pull request:
```bash
git push origin feature/your-feature-name
```

### Running Tests

```bash
python -m pytest tests/
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for the transformer models
- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search
- [Flask](https://flask.palletsprojects.com/) for the web framework 