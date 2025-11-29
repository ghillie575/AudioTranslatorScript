#!/usr/bin/env python3
"""
Local AI Audio Translator to Polish with Timestamp Preservation.
Uses local models only - NO cloud services by default, but falls back to
Google Translate (public endpoint) when Ollama fails.

- Whisper (faster-whisper) for transcription
- Ollama (local) for translation (preferred)
- Fallback to translate.googleapis.com when Ollama fails
- Coqui TTS (local) for generating translated audio
- **Feature added: Extract audio from VIDEO_FILE**
- **Feature added: Final outputs copied to a separate directory**

All processing done on your machine (except optional Google fallback).
"""

import os
import json
import subprocess
import traceback
from pathlib import Path
from datetime import datetime, timedelta
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from faster_whisper import WhisperModel
from tqdm import tqdm
import math
import tempfile
import shutil
from urllib.parse import quote_plus
try:
    from TTS.api import TTS
    from pydub import AudioSegment
    from pydub import effects
except ImportError:
    # Updated installation instruction
    print("❌ Please install Coqui TTS and pydub: pip install TTS[pytorch] pydub")
    exit(1)

#Settings
# --- Input/Output ---
INPUT_AUDIO = "audio.wav"         # Path to input audio file
OUTPUT_DIR = "translation_output" # Directory to store outputs
LANGUAGE = "Polish"               # Target language for translation
VIDEO_FILE = "video.mp4"
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


class LocalAudioTranslator:
    def __init__(self, input_file, output_dir, target_language="Polish"):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.target_language = target_language
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.llama_model = None
        self.llama_tokenizer = None
        # Create output directory structure
        self.dirs = {
            "root": self.output_dir,
            "audio": self.output_dir / "audio_files",
            "transcripts": self.output_dir / "transcripts",
            "translations": self.output_dir / "translations",
            "subtitles": self.output_dir / "subtitles",
            "tts": self.output_dir / "tts_output",
            "logs": self.output_dir / "logs",
            "models": self.output_dir / "models_cache",
            "final_output": self.output_dir / "final_translated_files" # ADDED final output dir
        }

        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        self.log_file = self.dirs["logs"] / f"translation_log_{self.timestamp}.txt"
        self.log("Local Audio Translator initialized")
        self.log(f"Device: {WHISPER_DEVICE}")

        # Initialize Whisper and TTS model placeholders
        self.whisper_model = None
        self.tts_engine = None

        # Paths for report generation and final outputs
        self.audio_file_path = None
        self.transcript_file_path = None
        self.translation_file_path = None
        self.srt_file_path = None
        self.vtt_file_path = None
        self.bilingual_file_path = None
        self.final_tts_file = None
        self.report_file_path = None

        # Check for existing files from previous runs
        self.existing_files = self._detect_existing_files()

    def _detect_existing_files(self):
        """Detect existing output files from previous runs"""
        existing = {
            "audio": None,
            "transcript": None,
            "translation": None,
            "subtitles": {"srt": None, "vtt": None, "bilingual": None},
            "report": None
        }

        audio_files = sorted(self.dirs["audio"].glob("extracted_audio_*.wav"), reverse=True)
        if audio_files:
            existing["audio"] = audio_files[0]

        transcript_files = sorted(self.dirs["transcripts"].glob("transcript_raw_*.json"), reverse=True)
        if transcript_files:
            existing["transcript"] = transcript_files[0]

        translation_files = sorted(self.dirs["translations"].glob("translations_*.json"), reverse=True)
        if translation_files:
            existing["translation"] = translation_files[0]

        srt_files = sorted(self.dirs["subtitles"].glob("subtitles_*.srt"), reverse=True)
        if srt_files:
            existing["subtitles"]["srt"] = srt_files[0]

        vtt_files = sorted(self.dirs["subtitles"].glob("subtitles_*.vtt"), reverse=True)
        if vtt_files:
            existing["subtitles"]["vtt"] = vtt_files[0]

        bilingual_files = sorted(self.dirs["subtitles"].glob("subtitles_bilingual_*.srt"), reverse=True)
        if bilingual_files:
            existing["subtitles"]["bilingual"] = bilingual_files[0]

        report_files = sorted(self.dirs["root"].glob("translation_report_*.txt"), reverse=True)
        if report_files:
            existing["report"] = report_files[0]

        return existing

    def log(self, message):
        """Log messages to file and console"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_message + "\n")

    def check_ollama_running(self):
        """Check if Ollama is running"""
        try:
            response = requests.get(OLLAMA_TAGS_URL, timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def check_ollama_model(self):
        """Check if the required model is available in Ollama"""
        try:
            response = requests.get(OLLAMA_TAGS_URL, timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", []) if isinstance(data, dict) else []
                model_names = [m.get("name") if isinstance(m, dict) else str(m) for m in models]
                # Check either full name or model family
                model_family = OLLAMA_MODEL.split(":")[0]
                return OLLAMA_MODEL in model_names or model_family in [m.split(":")[0] for m in model_names]
        except Exception:
            pass
        return False

    def load_whisper_model(self):
        """Load Whisper model locally"""
        if self.whisper_model is None:
            self.log(f"Loading Whisper {WHISPER_MODEL} model...")
            self.log("This may take a while on first run (downloading model)...")

            try:
                with tqdm(total=100, desc="Loading Whisper Model",
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
                    self.whisper_model = WhisperModel(
                        WHISPER_MODEL,
                        device=WHISPER_DEVICE,
                        compute_type=WHISPER_COMPUTE_TYPE,
                        download_root=str(self.dirs["models"])
                    )
                    pbar.update(100)
                self.log("Whisper model loaded successfully")
            except Exception as e:
                self.log(f"Error loading Whisper: {e}")
                raise

    def load_tts_model(self):
        """Load Coqui TTS model locally"""
        if self.tts_engine is None:
            self.log(f"Loading Coqui TTS model: {COQUI_TTS_MODEL}")
            try:
                # Load the TTS engine.
                self.tts_engine = TTS(COQUI_TTS_MODEL, gpu=torch.cuda.is_available())
                self.log("Coqui TTS model loaded successfully")
            except Exception as e:
                self.log(f"Error loading Coqui TTS: {e}")
                raise

    def extract_audio(self):
        """Extract audio from video if needed, convert to proper format with progress"""
        self.log("Step 1: Extracting/converting audio...")

        if self.existing_files["audio"] and self.existing_files["audio"].exists():
            self.log(f"⏭️  Using existing audio file: {self.existing_files['audio']}")
            self.audio_file_path = self.existing_files['audio']
            return self.audio_file_path

        if not self.input_file.exists():
            self.log(f"❌ Input file not found: {self.input_file}")
            raise FileNotFoundError(f"Input not found: {self.input_file}")

        self.audio_file_path = self.dirs["audio"] / f"extracted_audio_{self.timestamp}.wav"

        # Probe to get duration (for progress)
        try:
            probe_cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration,bit_rate",
                "-of", "default=noprint_wrappers=1",
                str(self.input_file)
            ]
            probe_output = subprocess.check_output(probe_cmd).decode().strip()
            duration = 0.0
            bitrate = 0
            for line in probe_output.splitlines():
                if line.startswith("duration="):
                    duration = float(line.split("=")[1])
                elif line.startswith("bit_rate="):
                    try:
                        bitrate = int(line.split("=")[1])
                    except:
                        bitrate = 0
            bitrate_kbps = bitrate / 1000 if bitrate else 0
        except Exception:
            duration = 0.0
            bitrate_kbps = 0

        cmd = [
            "ffmpeg", "-i", str(self.input_file),
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            "-y",
            "-progress", "pipe:1",
            "-loglevel", "error",
            str(self.audio_file_path)
        ]

        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, bufsize=1
            )

            # Use simpler bar_format that handles None values better
            if duration > 0:
                bar_format = '{desc}: {percentage:3.0f}%|{bar}| {n:.1f}/{total:.1f}s [{elapsed}<{remaining}]'
            else:
                bar_format = '{desc}: |{bar}| {n:.1f}s [{elapsed}]'
                process.wait()
                stderr_output = process.stderr.read()

            if process.returncode != 0:
                self.log(f"FFmpeg returned non-zero exit code. stderr: {stderr_output}")
                raise subprocess.CalledProcessError(process.returncode, cmd, output=stderr_output)

            if not self.audio_file_path.exists():
                raise RuntimeError("FFmpeg did not produce output audio file.")

            output_size_mb = os.path.getsize(self.audio_file_path) / (1024 * 1024)
            self.log(f"Audio extracted to: {self.audio_file_path} ({output_size_mb:.2f} MB)")
            return self.audio_file_path

        except subprocess.CalledProcessError as e:
            self.log(f"FFmpeg error: {e}")
            raise

    def transcribe_with_local_whisper(self, audio_file):
        """Transcribe audio using local Whisper model with progress"""
        self.log("Step 2: Transcribing audio with local Whisper...")

        self.transcript_file_path = self.dirs["transcripts"] / f"transcript_raw_{self.timestamp}.json"
        if self.existing_files["transcript"] and self.existing_files["transcript"].exists():
            self.log(f"⏭️  Using existing transcript: {self.existing_files['transcript']}")
            with open(self.existing_files["transcript"], "r", encoding="utf-8") as f:
                return json.load(f)

        self.load_whisper_model()

        # probe duration
        try:
            probe_cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_file)
            ]
            total_duration = float(subprocess.check_output(probe_cmd).decode().strip())
        except Exception:
            total_duration = 0.0

        audio_size_mb = os.path.getsize(audio_file) / (1024 * 1024)

        self.log(f"Transcribing {total_duration:.1f}s ({audio_size_mb:.1f} MB) audio file...")
        segments_list = []

        segments_generator, info = self.whisper_model.transcribe(
            str(audio_file),
            language=None,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )

        start_time = datetime.now()
        words_processed = 0
        chars_processed = 0
        last_end_time = 0.0

        with tqdm(total=int(total_duration) if total_duration > 0 else None,
                  desc="🎙️  Transcribing Audio", unit="s",
                  bar_format='{desc}: {percentage:3.0f}%|{bar}| {n:.1f}/{total:.1f}s [{elapsed}<{remaining}]') as pbar:

            for i, segment in enumerate(segments_generator):
                seg_dict = {
                    "id": i,
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": segment.text.strip(),
                    "words": []
                }

                words_in_segment = len(segment.text.strip().split())
                chars_in_segment = len(segment.text.strip())
                words_processed += words_in_segment
                chars_processed += chars_in_segment

                if getattr(segment, "words", None):
                    for word in segment.words:
                        seg_dict["words"].append({
                            "start": float(getattr(word, "start", 0.0)),
                            "end": float(getattr(word, "end", 0.0)),
                            "word": getattr(word, "word", "")
                        })

                segments_list.append(seg_dict)

                time_increment = segment.end - last_end_time
                if time_increment > 0 and total_duration > 0:
                    pbar.update(time_increment)
                last_end_time = segment.end

                # update postfix
                pbar.set_postfix({
                    'seg': f'{i+1}',
                    'words': words_processed,
                    'chars': chars_processed,
                    'audio': f'{segment.end:.1f}/{total_duration:.1f}s' if total_duration > 0 else f'{segment.end:.1f}s'
                })

        total_elapsed = (datetime.now() - start_time).total_seconds()
        rtf_final = total_elapsed / total_duration if total_duration > 0 else 0

        self.log(f"Transcription complete: {len(segments_list)} segments, {words_processed} words")
        transcript_data = {
            "language": getattr(info, "language", "unknown"),
            "language_probability": getattr(info, "language_probability", 0.0),
            "duration": getattr(info, "duration", total_duration),
            "segments": segments_list
        }

        with open(self.transcript_file_path, "w", encoding="utf-8") as f:
            json.dump(transcript_data, f, indent=2, ensure_ascii=False)

        # plain text
        text_file = self.dirs["transcripts"] / f"transcript_text_{self.timestamp}.txt"
        with open(text_file, "w", encoding="utf-8") as f:
            for seg in segments_list:
                f.write(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}\n")

        self.log(f"Text transcript saved to: {text_file}")
        return transcript_data

    def synthesize_tts_from_segments(self, translated_segments, audio_duration, out_filename=None):
        """
        Generate TTS with natural voice quality.
        Since translations are now duration-optimized, we can use natural timing.
        """
        self.log("Step 5: Generating single-file Polish TTS from translated segments...")

        if out_filename is None:
            out_filename = self.dirs["tts"] / f"tts_polish_{self.timestamp}.wav"
        else:
            out_filename = Path(out_filename)
        self.final_tts_file = out_filename

        self.load_tts_model()

        if not translated_segments:
            raise RuntimeError("No translated segments found for TTS generation.")

        tmpdir = Path(tempfile.mkdtemp(prefix="tts_tmp_"))
    
        total_segments = len(translated_segments)

        try:
            # --- Step 1: Generate TTS and analyze fit ---
            with tqdm(total=total_segments, desc="🎵 Generating Audio Segments", unit="seg",
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:

                segment_data = []
                fit_issues = 0
            
                for i, seg in enumerate(translated_segments):
                    text = seg.get('translated', '').strip()
                    start_sec = seg.get('start', 0.0)
                    end_sec = seg.get('end', start_sec + 1.0)
                    target_duration = end_sec - start_sec

                    if not text:
                        pbar.update(1)
                        continue

                    tmp_wav = tmpdir / f"sub_{i:04d}.wav"

                    # Generate TTS
                    self.tts_engine.tts_to_file(
                        text=text,
                        file_path=str(tmp_wav),
                        speaker_wav=COQUI_TTS_SPEAKER,
                        language=self._language_code_from_name(self.target_language)
                    )

                    # Check fit
                    seg_audio = AudioSegment.from_file(tmp_wav)
                    actual_duration = len(seg_audio) / 1000.0
                    duration_ratio = actual_duration / target_duration
                
                    if duration_ratio > 1.15:  # More than 15% over
                        fit_issues += 1
                        self.log(f"   ⚠ Segment {i+1}: TTS {actual_duration:.2f}s vs target {target_duration:.2f}s")
                        self.log(f"      Text: '{text[:60]}...'")

                    segment_data.append({
                        'start': start_sec,
                        'path': str(tmp_wav),
                        'actual_duration': actual_duration,
                        'target_duration': target_duration,
                        'fit_ratio': duration_ratio
                    })

                    pbar.update(1)

            if fit_issues > 0:
                self.log(f"   ⚠ {fit_issues}/{total_segments} segments don't fit well - consider re-translating with stricter constraints")

            # --- Step 2: Concatenate with natural timing (no speed changes) ---
            self.log("   └─ Concatenating audio segments with natural timing...")
            final_audio = AudioSegment.silent(duration=0)
            current_time_ms = 0

            for seg_info in segment_data:
                start_ms = int(round(seg_info['start'] * 1000))
                seg_audio = AudioSegment.from_file(seg_info['path'])

                # Add silence to reach start time
                if start_ms > current_time_ms:
                    gap_ms = start_ms - current_time_ms
                    final_audio += AudioSegment.silent(duration=gap_ms)
                    current_time_ms = start_ms

                # Append segment at natural length (NO speed modification)
                final_audio += seg_audio
                current_time_ms += len(seg_audio)

            # Add trailing silence
            if audio_duration > 0 and current_time_ms < audio_duration * 1000:
                trailing_ms = int(round(audio_duration * 1000 - current_time_ms))
                final_audio += AudioSegment.silent(duration=trailing_ms)

            # Export
            final_audio.export(str(out_filename), format="wav")
            self.log(f"Single TTS WAV saved to: {out_filename}")
            self.log(f"   Voice quality: Natural (no speed modifications applied)")
            return out_filename

        finally:
            try:
                self.log("   └─ Cleaning up temporary files...")
                shutil.rmtree(tmpdir)
            except Exception:
                pass


    def translate_with_google(self, text, target_lang):
        """Translate a single text snippet with Google Translate public endpoint (fallback).
        Returns translated text or raises Exception.
        """
        if not text:
            return ""
        # Build request
        params = {
            "client": "gtx",
            "sl": "auto",
            "tl": target_lang if target_lang else "en",
            "dt": "t",
            "q": text
        }
        # Use quote_plus to avoid very long URLs with problematic characters
        url = f"{GOOGLE_TRANSLATE_ENDPOINT}?client=gtx&sl=auto&tl={quote_plus(params['tl'])}&dt=t&q={quote_plus(text)}"
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                raise RuntimeError(f"Google translate endpoint returned {resp.status_code}")
            data = resp.json()
            # data[0] is list of translated chunks
            translated = "".join([chunk[0] for chunk in data[0] if chunk[0]])
            return translated
        except Exception as e:
            raise RuntimeError(f"Google fallback failed: {e}")
    
    def translate_with_ollama(self, transcript):
        """Translate transcript using local Ollama with duration-aware translation for dubbing."""
        self.log("Step 3: Translating with local Llama model via Ollama (dubbing-optimized)...")
        self.translation_file_path = self.dirs["translations"] / f"translations_{self.timestamp}.json"

        if self.existing_files["translation"] and self.existing_files["translation"].exists():
            self.log(f"⏭️  Using existing translation: {self.existing_files['translation']}")
            with open(self.existing_files["translation"], "r", encoding="utf-8") as f:
                return json.load(f)

        segments = transcript.get("segments", [])
        translated_segments = []
        batch_size = 5
        total_batches = (len(segments) - 1) // batch_size + 1 if segments else 0

        start_time = datetime.now()
        failed_batches = 0
        processed_words = 0
        processed_chars = 0

        with tqdm(total=len(segments), desc="🔄 Translating Segments", unit="seg",
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as main_pbar:

            for batch_idx in range(0, len(segments), batch_size):
                batch = segments[batch_idx:batch_idx + batch_size]
                current_batch = batch_idx // batch_size + 1

                # Prepare duration-aware prompt
                segments_text = ""
                for seg in batch:
                    duration = seg['end'] - seg['start']
                    word_count = len(seg['text'].split())
                    segments_text += f"[{seg['start']:.2f}s - {seg['end']:.2f}s] (Duration: {duration:.1f}s, Words: {word_count}) {seg['text']}\n"

                prompt = (
                    f"You are a professional dubbing translator for voice-over work. Translate these audio segments "
                    f"from {transcript.get('language', 'English')} to {self.target_language}.\n\n"
                    "CRITICAL DUBBING RULES:\n"
                    "1. **TIME CONSTRAINT**: Your translation MUST fit within the same duration as the original\n"
                    "2. **Syllable matching**: Match the syllable count as closely as possible to the original\n"
                    "3. **Natural speech**: Translation must sound natural when spoken at normal speed\n"
                    "4. **Adjust word count**: Use shorter/longer synonyms, remove filler words, or rephrase to fit time\n"
                    "5. **Meaning priority**: Preserve core meaning, but sacrifice literal accuracy for timing if needed\n"
                    "6. **Speech rate**: Assume normal speech rate (~150 words/minute in Polish, ~140 in English)\n\n"
                    "STRATEGY:\n"
                    "- If original has 10 words in 5 seconds → aim for 10-12 words in Polish\n"
                    "- For short segments (<3s): Keep it very concise, use contractions\n"
                    "- For long segments (>10s): You have more flexibility\n"
                    "- Remove unnecessary words: 'actually', 'basically', 'you know', etc.\n\n"
                    "Original segments (with timing info):\n"
                    f"{segments_text}\n\n"
                    "Return ONLY a JSON array with time-optimized translations:\n"
                    "[\n"
                    '  {"start": 0.00, "end": 5.50, "original": "original text", "translated": "concise fitted translation"},\n'
                    "  ...\n"
                    "]\n\n"
                    "REMEMBER: Shorter translations that fit naturally are better than accurate translations that sound rushed!\n\n"
                    "JSON array:"
            )

                # Attempt Ollama call
                success = False
                try:
                    payload = {
                        "model": OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.5,  # Slightly higher for more creative rephrasing
                            "top_p": 0.9,
                            "num_predict": 2048  # Allow longer responses for explanations
                        }
                    }
                
                    with tqdm(total=100, desc=f"   └─ API Call", unit="%", leave=False) as api_pbar:
                        api_pbar.update(10)
                        resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
                        api_pbar.update(40)
                    
                        if resp.status_code == 200:
                            result = resp.json()
                            response_text = None
                        
                            if isinstance(result, dict):
                                response_text = result.get("response") or result.get("text") or result.get("result") or None
                                if response_text is None:
                                    for v in result.values():
                                        if isinstance(v, str) and "[" in v and "]" in v:
                                            response_text = v
                                            break
                            elif isinstance(result, str):
                                response_text = result

                            if not response_text:
                                raise ValueError("Ollama response missing 'response' text")

                            # Extract JSON array
                            json_start = response_text.find("[")
                            json_end = response_text.rfind("]") + 1
                        
                            if json_start >= 0 and json_end > json_start:
                                json_str = response_text[json_start:json_end]
                                batch_translations = json.loads(json_str)
                                translated_segments.extend(batch_translations)
                                api_pbar.update(50)
                                success = True
                            else:
                                try:
                                    batch_translations = json.loads(response_text)
                                    if isinstance(batch_translations, list):
                                        translated_segments.extend(batch_translations)
                                        api_pbar.update(50)
                                        success = True
                                    else:
                                        raise ValueError("Parsed JSON is not a list")
                                except Exception:
                                    raise ValueError("No JSON array found in Ollama response")
                        else:
                            raise RuntimeError(f"Ollama API error: {resp.status_code} - {resp.text[:200]}")
                        
                except Exception as e:
                    failed_batches += 1
                    self.log(f"   ⚠ Ollama translation error in batch {current_batch}: {e}")
                    self.log("   → Trying Google fallback for this batch (if enabled)")

                if not success:
                    # Google fallback (won't be duration-aware, but better than nothing)
                    if USE_GOOGLE_FALLBACK:
                        for seg in batch:
                            try:
                                translated = self.translate_with_google(seg["text"], self._language_code_from_name(self.target_language))
                                translated_segments.append({
                                    "start": seg["start"],
                                    "end": seg["end"],
                                    "original": seg["text"],
                                    "translated": translated
                                })
                            except Exception as ge:
                                self.log(f"     ⚠ Google fallback failed for segment starting at {seg['start']}: {ge}")
                                translated_segments.append({
                                    "start": seg["start"],
                                    "end": seg["end"],
                                    "original": seg["text"],
                                    "translated": seg["text"]
                                })
                    else:
                        for seg in batch:
                            translated_segments.append({
                                "start": seg["start"],
                                "end": seg["end"],
                                "original": seg["text"],
                                "translated": seg["text"]
                            })

                processed_words += sum(len(seg['text'].split()) for seg in batch)
                processed_chars += sum(len(seg['text']) for seg in batch)
                main_pbar.update(len(batch))
                main_pbar.set_postfix({
                    'batch': f'{current_batch}/{total_batches}',
                    'words': processed_words,
                    'failed': failed_batches,
                    'status': '✓' if success else '⚠'
                })

        self.log(f"Translation complete: {len(translated_segments)} segments, {processed_words} words")
    
        # Save translations
        with open(self.translation_file_path, "w", encoding="utf-8") as f:
            json.dump(translated_segments, f, indent=2, ensure_ascii=False)

        # Save text with timing analysis
        text_file = self.dirs["translations"] / f"translations_text_{self.timestamp}.txt"
        with open(text_file, "w", encoding="utf-8") as f:
            for seg in translated_segments:
                duration = seg['end'] - seg['start']
                orig_words = len(seg['original'].split())
                trans_words = len(seg['translated'].split())
            
                f.write(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] (Duration: {duration:.1f}s)\n")
                f.write(f"Original ({orig_words} words):   {seg['original']}\n")
                f.write(f"Translated ({trans_words} words): {seg['translated']}\n")
                f.write(f"Word ratio: {trans_words/orig_words:.2f}x\n\n")

        return translated_segments

    def _language_code_from_name(self, name):
        """Basic mapping from language name to language code for Google fallback and TTS."""
        mapping = {
            "Polish": "pl",
            "English": "en",
            "Spanish": "es",
            "French": "fr",
            "German": "de",
            # add more as needed
        }
        return mapping.get(name, "en")

    def _format_timestamp(self, seconds, format_type='srt'):
        """Format seconds (float) into SRT (00:00:00,000) or VTT (00:00:00.000)."""
        # handle negative or None
        try:
            seconds = float(seconds)
        except (TypeError, ValueError):
            seconds = 0.0
        
        if seconds < 0: seconds = 0.0

        td = timedelta(seconds=seconds)
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds_int = divmod(remainder, 60)
        milliseconds = td.microseconds // 1000

        if format_type == 'srt':
            # SRT format: HH:MM:SS,mmm
            return f"{hours:02}:{minutes:02}:{seconds_int:02},{milliseconds:03}"
        elif format_type == 'vtt':
            # VTT format: HH:MM:SS.mmm
            return f"{hours:02}:{minutes:02}:{seconds_int:02}.{milliseconds:03}"
        else:
            raise ValueError("Invalid format_type")
        
    def generate_subtitles(self, original_segments, translated_segments):
        """Step 4: Generate SRT and VTT subtitle files (translated and bilingual)"""
        self.log("Step 4: Generating subtitle files (.srt, .vtt)...")

        self.srt_file_path = self.dirs["subtitles"] / f"subtitles_{self.timestamp}.srt"
        self.vtt_file_path = self.dirs["subtitles"] / f"subtitles_{self.timestamp}.vtt"
        self.bilingual_file_path = self.dirs["subtitles"] / f"subtitles_bilingual_{self.timestamp}.srt"
        
        # Merge data (ensure lengths match, although they should if done correctly)
        min_len = min(len(original_segments), len(translated_segments))
        merged_segments = []
        for i in range(min_len):
            original = original_segments[i]
            translated = translated_segments[i]
            
            # Simple check for structure integrity
            if 'start' in original and 'end' in original and 'translated' in translated:
                merged_segments.append({
                    "id": i + 1,
                    "start": translated['start'],
                    "end": translated['end'],
                    "original": original['text'],
                    "translated": translated['translated']
                })
        
        if not merged_segments:
            self.log("❌ Subtitle generation failed: No segments to merge.")
            return

        # Write Translated SRT and VTT
        with open(self.srt_file_path, "w", encoding="utf-8") as srt_f, \
             open(self.vtt_file_path, "w", encoding="utf-8") as vtt_f:
            
            # VTT Header
            vtt_f.write("WEBVTT\n\n")

            for seg in merged_segments:
                srt_start = self._format_timestamp(seg['start'], 'srt')
                srt_end = self._format_timestamp(seg['end'], 'srt')
                vtt_start = self._format_timestamp(seg['start'], 'vtt')
                vtt_end = self._format_timestamp(seg['end'], 'vtt')

                # SRT
                srt_f.write(f"{seg['id']}\n")
                srt_f.write(f"{srt_start} --> {srt_end}\n")
                srt_f.write(f"{seg['translated']}\n\n")

                # VTT
                vtt_f.write(f"{vtt_start} --> {vtt_end}\n")
                vtt_f.write(f"{seg['translated']}\n\n")

        self.log(f"Translated subtitles saved to: {self.srt_file_path} and {self.vtt_file_path}")

        # Write Bilingual SRT
        with open(self.bilingual_file_path, "w", encoding="utf-8") as bilin_f:
            for seg in merged_segments:
                srt_start = self._format_timestamp(seg['start'], 'srt')
                srt_end = self._format_timestamp(seg['end'], 'srt')
                
                bilin_f.write(f"{seg['id']}\n")
                bilin_f.write(f"{srt_start} --> {srt_end}\n")
                bilin_f.write(f"Original: {seg['original']}\n")
                bilin_f.write(f"Translated: {seg['translated']}\n\n")
        
        self.log(f"Bilingual subtitles saved to: {self.bilingual_file_path}")
        
    def copy_final_outputs(self):
        """Copy the final SRT, VTT, and TTS WAV files to the dedicated output directory."""
        self.log("Step 6: Copying final outputs to dedicated directory...")
        
        final_dir = self.dirs["final_output"]
        copied_files = []
        
        # Files to copy (only the latest ones generated in this run)
        files_to_check = [
            self.srt_file_path,
            self.vtt_file_path,
            self.bilingual_file_path,
            self.final_tts_file
        ]
        
        for file_path in files_to_check:
            if file_path and Path(file_path).exists():
                try:
                    dest_path = final_dir / file_path.name
                    shutil.copy2(file_path, dest_path)
                    copied_files.append(dest_path)
                except Exception as e:
                    self.log(f"❌ Error copying {file_path.name}: {e}")

        self.log(f"Successfully copied {len(copied_files)} files to: {final_dir}")
        for f in copied_files:
            self.log(f"   └─ {f.name}")
        
    def generate_report(self):
        """Generate a final summary report."""
        self.log("Step 7: Generating final report...")
        self.report_file_path = self.output_dir / f"translation_report_{self.timestamp}.txt"
        
        # Get duration for RTF calculation if available
        duration_sec = 0.0
        try:
            if self.audio_file_path and self.audio_file_path.exists():
                probe_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(self.audio_file_path)]
                duration_sec = float(subprocess.check_output(probe_cmd).decode().strip())
        except Exception:
            duration_sec = 0.0

        # Gather file paths
        audio_filename = self.audio_file_path.name if self.audio_file_path else "N/A"
        transcript_filename = self.transcript_file_path.name if self.transcript_file_path else "N/A"
        translation_filename = self.translation_file_path.name if self.translation_file_path else "N/A"
        srt_filename = self.srt_file_path.name if self.srt_file_path else "N/A"
        vtt_filename = self.vtt_file_path.name if self.vtt_file_path else "N/A"
        bilingual_filename = self.bilingual_file_path.name if self.bilingual_file_path else "N/A"
        tts_filename = self.final_tts_file.name if self.final_tts_file else "N/A"
        
        # Read log for overall timing (simple approximation)
        try:
            start_time_log = None
            end_time_log = None
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if "initialized" in line:
                        try:
                            start_time_log = datetime.strptime(line.split(']')[0].strip('[')[:-3], "%Y-%m-%d %H:%M:%S")
                        except ValueError:
                            pass
                    if "Finished translation pipeline" in line:
                        try:
                            end_time_log = datetime.strptime(line.split(']')[0].strip('[')[:-3], "%Y-%m-%d %H:%M:%S")
                        except ValueError:
                            pass
            
            total_time_seconds = (end_time_log - start_time_log).total_seconds() if start_time_log and end_time_log else 0
            
            rtf_overall = total_time_seconds / duration_sec if duration_sec > 0 and total_time_seconds > 0 else 0
        except Exception:
            total_time_seconds = 0
            rtf_overall = 0
            
        
        with open(self.report_file_path, "w", encoding="utf-8") as f:
            f.write(f"--- Translation Report - {self.timestamp} ---\n\n")
            f.write(f"Input File: {self.input_file.name}\n")
            f.write(f"Target Language: {self.target_language}\n")
            f.write(f"Output Directory: {self.output_dir.resolve()}\n")
            f.write(f"Video Duration: {duration_sec:.2f} seconds\n")
            f.write(f"Overall Processing Time: {total_time_seconds:.2f} seconds\n")
            f.write(f"Real-Time Factor (RTF): {rtf_overall:.2f}x (Lower is better)\n\n")
            
            f.write("--- Generated Files ---\n")
            f.write(f"Extracted Audio: {audio_filename}\n")
            f.write(f"Raw Transcript (JSON): {transcript_filename}\n")
            f.write(f"Translation Data (JSON): {translation_filename}\n")
            f.write(f"Translated Subtitle (SRT): {srt_filename}\n")
            f.write(f"Translated Subtitle (VTT): {vtt_filename}\n")
            f.write(f"Bilingual Subtitle (SRT): {bilingual_filename}\n")
            f.write(f"Final TTS Audio: {tts_filename}\n")
            f.write(f"Full Log: {self.log_file.name}\n\n")
            
            f.write(f"Final Outputs Copied To: {self.dirs['final_output'].resolve()}\n")

        self.log(f"Final report generated: {self.report_file_path}")
        
    def run_pipeline(self):
        """Main method to run the entire translation pipeline."""
        try:
            # 1. Extract Audio from VIDEO_FILE
            audio_file = self.extract_audio()

            # 2. Transcribe
            transcript_data = self.transcribe_with_local_whisper(audio_file)
            original_segments = transcript_data.get("segments", [])
            audio_duration = transcript_data.get("duration", 0.0)

            # 3. Translate
            translated_segments = self.translate_with_ollama(transcript_data)

            # 4. Generate Subtitles
            self.generate_subtitles(original_segments, translated_segments)

            # 5. Generate TTS Audio
            if COQUI_TTS_MODEL and COQUI_TTS_SPEAKER:
                self.synthesize_tts_from_segments(translated_segments, audio_duration)
            else:
                self.log("⚠️ Skipping TTS generation: COQUI_TTS_MODEL or COQUI_TTS_SPEAKER not configured.")
            
            # 6. Copy final outputs
            self.copy_final_outputs()

            # 7. Generate Report
            self.generate_report()

            self.log("\n✅ Finished translation pipeline successfully!")

        except Exception as e:
            self.log(f"\n❌ Pipeline failed: {e}")
            self.log(traceback.format_exc())
            self.generate_report() # Generate report even on failure

# Main execution block
if __name__ == "__main__":
    # Ensure VIDEO_FILE is defined in settings.py
    if 'VIDEO_FILE' not in globals() or not VIDEO_FILE:
        print("❌ Error: VIDEO_FILE not defined in settings.py.")
        exit(1)
        
    if 'OUTPUT_DIR' not in globals() or not OUTPUT_DIR:
        print("❌ Error: OUTPUT_DIR not defined in settings.py.")
        exit(1)
    # Initialize and run the translator
    translator = LocalAudioTranslator(
        input_file=VIDEO_FILE, 
        output_dir=OUTPUT_DIR, 
        target_language=LANGUAGE
    )
    
    # Check dependencies before starting the whole process
    try:
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            print("❌ FFmpeg/FFprobe not found. Please install it and ensure it's in your PATH.")
            exit(1)
    except Exception as e:
        print(f"❌ Error checking FFmpeg: {e}")
        exit(1)

    translator.run_pipeline()