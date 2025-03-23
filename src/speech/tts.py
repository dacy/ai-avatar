"""
Text-to-Speech (TTS) module for converting text to speech.
"""

import os
import tempfile
import numpy as np
from typing import Union, Dict, Any, Optional, Tuple
from loguru import logger
import soundfile as sf
import uuid

class TTSProcessor:
    """
    Text-to-Speech processor for converting text to speech.
    Supports multiple TTS engines (Piper, Coqui).
    """
    
    def __init__(self, engine: str = "piper", voice: str = "en_US-amy-medium", speed: float = 1.0):
        """
        Initialize the TTS processor.
        
        Args:
            engine: TTS engine to use ("piper" or "coqui")
            voice: Voice model to use
            speed: Speech rate multiplier
        """
        self.engine = engine.lower()
        self.voice = voice
        self.speed = speed
        self.model = None
        
        if self.engine == "piper":
            self._init_piper()
        elif self.engine == "coqui":
            self._init_coqui()
        else:
            raise ValueError(f"Unsupported TTS engine: {engine}")
    
    def _init_piper(self):
        """Initialize the Piper TTS model"""
        try:
            from piper import PiperVoice
            
            # Check if voice exists, download if not
            voice_dir = os.path.join(os.path.expanduser("~"), ".local", "share", "piper", "voices")
            os.makedirs(voice_dir, exist_ok=True)
            
            voice_path = os.path.join(voice_dir, f"{self.voice}.onnx")
            config_path = os.path.join(voice_dir, f"{self.voice}.json")
            
            if not os.path.exists(voice_path) or not os.path.exists(config_path):
                logger.info(f"Piper voice not found: {self.voice}, downloading...")
                # In a real implementation, we would download the voice here
                # For now, we'll just raise an error
                raise FileNotFoundError(f"Piper voice not found: {self.voice}")
            
            logger.info(f"Loading Piper voice: {self.voice}")
            self.model = PiperVoice.load(voice_path, config_path)
            logger.info("Piper voice loaded successfully")
            
        except ImportError:
            logger.error("Piper is not installed. Please install it with 'pip install piper-tts'")
            raise
    
    def _init_coqui(self):
        """Initialize the Coqui TTS model"""
        try:
            import TTS
            from TTS.utils.manage import ModelManager
            from TTS.utils.synthesizer import Synthesizer
            
            # Get the model manager
            model_manager = ModelManager()
            
            # Find a suitable model based on the voice
            language = self.voice.split("-")[0]
            model_path, config_path, model_item = model_manager.download_model(language)
            
            logger.info(f"Loading Coqui TTS model: {model_item['name']}")
            self.model = Synthesizer(
                model_path=model_path,
                config_path=config_path,
                use_cuda=False  # Set to True if using GPU
            )
            logger.info("Coqui TTS model loaded successfully")
            
        except ImportError:
            logger.error("Coqui TTS is not installed. Please install it with 'pip install TTS'")
            raise
    
    def synthesize_text(self, text: str) -> Tuple[np.ndarray, int]:
        """
        Synthesize text to speech.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        if self.engine == "piper":
            return self._synthesize_piper(text)
        elif self.engine == "coqui":
            return self._synthesize_coqui(text)
        else:
            raise ValueError(f"Unsupported TTS engine: {self.engine}")
    
    def synthesize_to_file(self, text: str, output_path: str) -> str:
        """
        Synthesize text to speech and save to a file.
        
        Args:
            text: Text to synthesize
            output_path: Path to save the audio file
            
        Returns:
            Path to the saved audio file
        """
        # Synthesize the text
        audio_data, sample_rate = self.synthesize_text(text)
        
        # Save the audio to a file
        sf.write(output_path, audio_data, sample_rate)
        
        logger.info(f"Synthesized audio saved to: {output_path}")
        
        return output_path
    
    def synthesize_to_temp_file(self, text: str, suffix: str = ".wav") -> str:
        """
        Synthesize text to speech and save to a temporary file.
        
        Args:
            text: Text to synthesize
            suffix: File suffix
            
        Returns:
            Path to the temporary audio file
        """
        # Create a temporary file
        temp_dir = os.path.join(tempfile.gettempdir(), "ai_avatar_tts")
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}{suffix}")
        
        # Synthesize to the temporary file
        self.synthesize_to_file(text, temp_path)
        
        return temp_path
    
    def _synthesize_piper(self, text: str) -> Tuple[np.ndarray, int]:
        """
        Synthesize text using Piper TTS.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        logger.info(f"Synthesizing text with Piper: {text[:50]}...")
        
        # Synthesize the text
        audio_data, sample_rate = self.model.synthesize(text)
        
        # Apply speed adjustment if needed
        if self.speed != 1.0:
            # This is a simple implementation - in a real system, we would use a proper
            # time-stretching algorithm that preserves pitch
            import librosa
            audio_data = librosa.effects.time_stretch(audio_data, rate=self.speed)
        
        logger.info(f"Synthesis complete: {len(audio_data)} samples at {sample_rate} Hz")
        
        return audio_data, sample_rate
    
    def _synthesize_coqui(self, text: str) -> Tuple[np.ndarray, int]:
        """
        Synthesize text using Coqui TTS.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        logger.info(f"Synthesizing text with Coqui: {text[:50]}...")
        
        # Synthesize the text
        audio_data = self.model.tts(text, speed=self.speed)
        sample_rate = self.model.output_sample_rate
        
        logger.info(f"Synthesis complete: {len(audio_data)} samples at {sample_rate} Hz")
        
        return audio_data, sample_rate 