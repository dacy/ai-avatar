"""
Automatic Speech Recognition (ASR) module for converting speech to text.
"""

import os
import tempfile
import numpy as np
from typing import Union, Dict, Any, Optional
from loguru import logger
import whisper
import soundfile as sf

class ASRProcessor:
    """
    Automatic Speech Recognition processor for converting speech to text.
    Supports multiple ASR engines (Whisper, Vosk).
    """
    
    def __init__(self, engine: str = "whisper", model_name: str = "small", language: str = "en"):
        """
        Initialize the ASR processor.
        
        Args:
            engine: ASR engine to use ("whisper" or "vosk")
            model_name: Model name/size to use
            language: Language code
        """
        self.engine = engine.lower()
        self.model_name = model_name
        self.language = language
        self.model = None
        
        if self.engine == "whisper":
            self._init_whisper()
        elif self.engine == "vosk":
            self._init_vosk()
        else:
            raise ValueError(f"Unsupported ASR engine: {engine}")
    
    def _init_whisper(self):
        """Initialize the Whisper ASR model"""
        logger.info(f"Loading Whisper model: {self.model_name}")
        self.model = whisper.load_model(self.model_name)
        logger.info("Whisper model loaded successfully")
    
    def _init_vosk(self):
        """Initialize the Vosk ASR model"""
        try:
            from vosk import Model, KaldiRecognizer
            
            # Check if model exists, download if not
            model_path = os.path.join(os.path.expanduser("~"), ".cache", "vosk", f"vosk-model-{self.model_name}-{self.language}")
            if not os.path.exists(model_path):
                logger.info(f"Vosk model not found at {model_path}, downloading...")
                # In a real implementation, we would download the model here
                # For now, we'll just raise an error
                raise FileNotFoundError(f"Vosk model not found: {model_path}")
            
            logger.info(f"Loading Vosk model from {model_path}")
            self.model = Model(model_path)
            logger.info("Vosk model loaded successfully")
            
        except ImportError:
            logger.error("Vosk is not installed. Please install it with 'pip install vosk'")
            raise
    
    def transcribe_audio_file(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe an audio file to text.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary containing the transcription and metadata
        """
        if self.engine == "whisper":
            return self._transcribe_whisper(audio_path)
        elif self.engine == "vosk":
            return self._transcribe_vosk(audio_path)
        else:
            raise ValueError(f"Unsupported ASR engine: {self.engine}")
    
    def transcribe_audio_data(self, audio_data: Union[bytes, np.ndarray], sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: Audio data as bytes or numpy array
            sample_rate: Sample rate of the audio data
            
        Returns:
            Dictionary containing the transcription and metadata
        """
        # Save audio data to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            
            # Convert audio data to the right format
            if isinstance(audio_data, bytes):
                temp_file.write(audio_data)
            else:
                # Assume numpy array
                sf.write(temp_path, audio_data, sample_rate)
        
        try:
            # Transcribe the temporary file
            result = self.transcribe_audio_file(temp_path)
            return result
        finally:
            # Clean up the temporary file
            os.unlink(temp_path)
    
    def _transcribe_whisper(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe an audio file using Whisper.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary containing the transcription and metadata
        """
        logger.info(f"Transcribing audio file with Whisper: {audio_path}")
        
        # Transcribe the audio
        result = self.model.transcribe(
            audio_path,
            language=self.language,
            fp16=False  # Set to True if using GPU
        )
        
        # Extract the text
        text = result["text"].strip()
        
        logger.info(f"Transcription complete: {text[:50]}...")
        
        return {
            "text": text,
            "segments": result.get("segments", []),
            "language": result.get("language", self.language)
        }
    
    def _transcribe_vosk(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe an audio file using Vosk.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary containing the transcription and metadata
        """
        try:
            from vosk import KaldiRecognizer
            import wave
            import json
            
            logger.info(f"Transcribing audio file with Vosk: {audio_path}")
            
            # Open the audio file
            wf = wave.open(audio_path, "rb")
            
            # Check if the audio format is compatible with Vosk
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
                logger.warning("Audio file format is not ideal for Vosk (should be mono PCM)")
            
            # Create a recognizer
            rec = KaldiRecognizer(self.model, wf.getframerate())
            rec.SetWords(True)
            
            # Process the audio in chunks
            results = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    part_result = json.loads(rec.Result())
                    results.append(part_result)
            
            # Get the final result
            part_result = json.loads(rec.FinalResult())
            results.append(part_result)
            
            # Combine all results
            text = " ".join(r.get("text", "") for r in results if "text" in r)
            
            logger.info(f"Transcription complete: {text[:50]}...")
            
            return {
                "text": text,
                "results": results
            }
            
        except ImportError:
            logger.error("Vosk is not installed. Please install it with 'pip install vosk'")
            raise 