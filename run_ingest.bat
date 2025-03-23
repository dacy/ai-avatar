@echo off
echo Running document ingestion for Voice Search...
cd %~dp0
call venv\Scripts\activate
python scripts/voice_search_cli.py ingest --config config/config.yaml --input-dir data/confluence_pages --output-index data/vectors/voice_search_index.faiss
if %ERRORLEVEL% NEQ 0 (
  echo Ingestion failed with error code %ERRORLEVEL%
  echo Detailed error information:
  type nul > error_log.txt
  python scripts/voice_search_cli.py ingest --config config/config.yaml --input-dir data/confluence_pages --output-index data/vectors/voice_search_index.faiss > error_log.txt 2>&1
  type error_log.txt
) else (
  echo Ingestion completed successfully!
)
pause 