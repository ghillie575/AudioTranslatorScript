# -*- coding: utf-8 -*-
"""
Configuration for Local AI Audio Translator
"""

from pathlib import Path
import torch

# --- Input/Output ---
INPUT_AUDIO = "audio.wav"         # Path to input audio file
OUTPUT_DIR = "translation_output" # Directory to store outputs
LANGUAGE = "Polish"               # Target language for translation

# --- Whisper Model Settings ---
WHISPER_MODEL = "large-v3"                 # Options: tiny, base, small, medium, large-v2, large-v3
WHISPER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"

# --- Coqui TTS Settings ---
COQUI_TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
COQUI_TTS_SPEAKER = "sample.wav"  # Path to speaker sample audio

# --- Ollama Settings ---
OLLAMA_MODEL = "llama3.1:8b"  # or llama3.1:70b
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"

# --- Google Translate Fallback ---
USE_GOOGLE_FALLBACK = True
GOOGLE_TRANSLATE_ENDPOINT = "https://translate.googleapis.com/translate_a/single"