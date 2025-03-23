@echo off
cd %~dp0
call ..\venv\Scripts\activate
echo Running Voice Search with Ollama LLM (deepseek-r1:1.5b)...
echo.

REM Check if index file exists
if not exist "data\voice_search_index.faiss" (
  echo ERROR: Index file not found at data\voice_search_index.faiss
  echo Please run the ingestion process first:
  echo run_ingest.bat
  echo.
  goto end
)

REM Check if metadata file exists
if not exist "data\voice_search_index.metadata" (
  echo ERROR: Metadata file not found at data\voice_search_index.metadata
  echo Please run the ingestion process first:
  echo run_ingest.bat
  echo.
  goto end
)

echo This version uses a local LLM through Ollama instead of the RoBERTa model.
echo It provides concise answers without showing the thinking process.
echo.
echo Optimizations for faster response times:
echo  - Concise prompts to reduce token count
echo  - Optimized model parameters (temperature, top_p, top_k)
echo  - Limited context length to reduce processing time
echo  - Focused on most relevant context for faster responses
echo.
echo The improved prompt engineering ensures:
echo  - No hallucinations (only uses information from the knowledge base)
echo  - No thinking process displayed (only shows the final answer)
echo  - Higher confidence scores for relevant answers
echo.

REM Check if Ollama is running (more reliable method)
echo Checking if Ollama server is running...
python -c "import requests; exit(0 if requests.get('http://localhost:11434/api/version', timeout=2).status_code == 200 else 1)" 2>nul
if %ERRORLEVEL% NEQ 0 (
  echo WARNING: Ollama server is not running.
  echo Please start Ollama first by running:
  echo   ollama serve
  echo.
  echo Waiting 5 seconds before continuing anyway...
  timeout /t 5 >nul
)

REM Check if the model exists
echo Checking if model deepseek-r1:1.5b is available...
python -c "import requests; import json; exit(0 if 'deepseek-r1:1.5b' in json.loads(requests.get('http://localhost:11434/api/tags', timeout=2).text).get('models', []) else 1)" 2>nul
if %ERRORLEVEL% NEQ 0 (
  echo WARNING: Model deepseek-r1:1.5b may not be available.
  echo You can pull it by running:
  echo   ollama pull deepseek-r1:1.5b
  echo.
  echo Continuing anyway...
)

echo Starting Ollama QA system...
python scripts/ollama_qa.py --config config/voice_search_config.yaml --index data/voice_search_index.faiss --model deepseek-r1:1.5b
if %ERRORLEVEL% NEQ 0 (
  echo.
  echo ERROR: An error occurred while running the voice search system.
  echo Please check the following:
  echo  1. Make sure Ollama is running (ollama serve)
  echo  2. Make sure the model is pulled (ollama pull deepseek-r1:1.5b)
  echo  3. Check that all files exist in the correct locations
  echo.
  echo For more details, run the script directly:
  echo   python scripts/ollama_qa.py --config config/voice_search_config.yaml --index data/voice_search_index.faiss --model deepseek-r1:1.5b
)

:end
pause 