#!/usr/bin/env python3
"""
Local AI Audio Translator to Polish with Timestamp Preservation.
Uses local models only - NO cloud services by default, but falls back to
Google Translate (public endpoint) when Ollama fails.

- Whisper (faster-whisper) for transcription
- Ollama (local) for translation (preferred)
- Fallback to translate.googleapis.com when Ollama fails
- Coqui TTS (local) for generating translated audio

All processing done on your machine (except optional Google fallback).
"""

import os
import json
import subprocess
import traceback
from pathlib import Path
from datetime import datetime, timedelta
import requests
import torch
from faster_whisper import WhisperModel
from tqdm import tqdm
import math
import tempfile # Added for temporary file management in TTS
import shutil   # Added for temporary directory cleanup in TTS
from urllib.parse import quote_plus
try:
    from TTS.api import TTS
    from pydub import AudioSegment # Added for audio concatenation and silence
except ImportError:
    # Updated installation instruction
    print("❌ Please install Coqui TTS and pydub: pip install TTS[pytorch] pydub")
    exit(1)
# Load settings
from settings import *


class LocalAudioTranslator:
    def __init__(self, input_file, output_dir, target_language="Polish"):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.target_language = target_language
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create output directory structure
        self.dirs = {
            "root": self.output_dir,
            "audio": self.output_dir / "audio_files",
            "transcripts": self.output_dir / "transcripts",
            "translations": self.output_dir / "translations",
            "subtitles": self.output_dir / "subtitles",
            "tts": self.output_dir / "tts_output", # Added TTS output directory
            "logs": self.output_dir / "logs",
            "models": self.output_dir / "models_cache"
        }

        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        self.log_file = self.dirs["logs"] / f"translation_log_{self.timestamp}.txt"
        self.log("Local Audio Translator initialized")
        self.log(f"Device: {WHISPER_DEVICE}")

        # Initialize Whisper and TTS model placeholders
        self.whisper_model = None
        self.tts_engine = None # Added TTS engine placeholder

        # Paths for report generation
        self.audio_file_path = None
        self.transcript_file_path = None
        self.translation_file_path = None
        self.srt_file_path = None
        self.vtt_file_path = None
        self.bilingual_file_path = None
        self.final_tts_file = None # Added path for final TTS file
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
                # Load the TTS engine. The model name has been updated to fix the KeyError.
                self.tts_engine = TTS(COQUI_TTS_MODEL, gpu=torch.cuda.is_available())
                self.log("Coqui TTS model loaded successfully")
            except Exception as e:
                self.log(f"Error loading Coqui TTS: {e}")
                raise

    def extract_audio(self):
        """Extract audio from video if needed, convert to proper format with progress"""
        self.log("Step 1: Extracting/converting audio...")

        if self.existing_files["audio"] and self.existing_files["audio"].exists():
            self.log(f"⏭️  Using existing audio file: {self.existing_files['audio']}")
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

            with tqdm(total=int(duration) if duration > 0 else None,
                      desc="🔊 Extracting Audio", unit="s",
                      bar_format='{desc}: {percentage:3.0f}%|{bar}| {n:.1f}/{total:.1f}s [{elapsed}<{remaining}]') as pbar:

                last_time = 0.0
                for line in process.stdout:
                    line = line.strip()
                    if not line:
                        continue
                    # ffmpeg -progress prints lines like key=value
                    if "=" in line:
                        key, val = line.split("=", 1)
                        if key == "out_time_ms":
                            try:
                                t_sec = int(val) / 1_000_000.0
                                if duration > 0:
                                    to_add = max(0.0, t_sec - last_time)
                                    pbar.update(to_add)
                                last_time = t_sec
                            except:
                                pass
                        elif key == "progress" and val == "end":
                            # finish
                            if duration > 0:
                                pbar.update(max(0, int(duration) - int(last_time)))
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
            self.log(f"⏭️  Using existing transcript: {self.existing_files['transcript']}")
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
                  desc="🎙️  Transcribing Audio", unit="s",
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
        Generate a single TTS WAV directly from translated segments (JSON data) with progress.
        This function replaces the old subtitle-based generation.
        """
        self.log("Step 5: Generating single-file Polish TTS from translated segments...")
        
        if out_filename is None:
            out_filename = self.dirs["tts"] / f"tts_polish_{self.timestamp}.wav"
        else:
            out_filename = Path(out_filename)
        self.final_tts_file = out_filename
        
        self.load_tts_model() # Load TTS model (model string is now fixed)

        if not translated_segments:
            raise RuntimeError("No translated segments found for TTS generation.")

        tmpdir = Path(tempfile.mkdtemp(prefix="tts_tmp_"))
        tmp_files = [] # Stores (start_sec, tmp_wav_path)
        
        total_segments = len(translated_segments)

        try:
            # Progress bar for segment generation
            with tqdm(total=total_segments, desc="🎵 Generating Audio Segments", unit="seg",
                      bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:

                for i, seg in enumerate(translated_segments):
                    text = seg.get('translated', '').strip()
                    start_sec = seg.get('start', 0.0)
                    
                    if not text:
                        pbar.update(1)
                        continue
                        
                    tmp_wav = tmpdir / f"sub_{i:04d}.wav"
                    
                    # Generate TTS audio for the segment
                    self.tts_engine.tts_to_file(
                        text=text, 
                        file_path=str(tmp_wav),
                        speaker_wav=COQUI_TTS_SPEAKER, # <--- CRITICAL: MUST BE VALID PATH
                        language=self._language_code_from_name(self.target_language)
                    )
                    
                    tmp_files.append((start_sec, str(tmp_wav)))
                    pbar.update(1)

            # Concatenate audio segments, adding silence/gaps
            self.log("   └─ Concatenating audio segments...")
            tmp_files.sort(key=lambda x: x[0])
            final_audio = AudioSegment.silent(duration=0) # Start with empty audio
            current_time_ms = 0

            for start_sec, wav_path in tmp_files:
                start_ms = int(round(float(start_sec) * 1000))
                seg_audio = AudioSegment.from_file(wav_path)
                seg_duration_ms = len(seg_audio)

                # Add silence gap
                if start_ms > current_time_ms:
                    gap_ms = start_ms - current_time_ms
                    final_audio += AudioSegment.silent(duration=gap_ms)
                    current_time_ms += gap_ms
                
                # Append translated segment audio
                final_audio += seg_audio
                current_time_ms += seg_duration_ms

            # Add trailing silence if the final audio is shorter than the original
            if audio_duration > 0 and current_time_ms < audio_duration * 1000:
                 trailing_gap_ms = int(round(audio_duration * 1000 - current_time_ms))
                 final_audio += AudioSegment.silent(duration=trailing_gap_ms)


            final_audio.export(str(out_filename), format="wav")
            self.log(f"Single TTS WAV saved to: {out_filename}")
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
        """Translate transcript using local Ollama with Google fallback if necessary."""
        self.log("Step 3: Translating with local Llama model via Ollama...")
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
        total_tokens_generated = 0

        with tqdm(total=len(segments), desc="🔄 Translating Segments", unit="seg",
                  bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as main_pbar:

            for batch_idx in range(0, len(segments), batch_size):
                batch = segments[batch_idx:batch_idx + batch_size]
                current_batch = batch_idx // batch_size + 1

                # Prepare prompt
                segments_text = ""
                for seg in batch:
                    segments_text += f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}\n"

                prompt = (
                    f"You are a professional translator. Translate the following timestamped audio transcript segments "
                    f"from {transcript.get('language', 'English')} to {self.target_language}.\n\n"
                    "CRITICAL REQUIREMENTS:\n"
                    "1. Translate naturally - maintain the same meaning and tone\n"
                    "2. Keep translations concise to fit the same time duration\n"
                    "3. Preserve technical terms appropriately\n"
                    "4. Use natural phrasing\n"
                    "5. Maintain emotional intensity and style\n\n"
                    "Transcript segments:\n"
                    f"{segments_text}\n\n"
                    "Return ONLY a JSON array with this exact structure (no other text):\n"
                    "[\n  {\"start\": 0.00, \"end\": 5.50, \"original\": \"original text\", \"translated\": \"translated text\"},\n  ...\n]\n\nJSON array:"
                )

                # Attempt Ollama call
                success = False
                try:
                    # Call Ollama
                    payload = {
                        "model": OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "top_p": 0.9
                        }
                    }
                    api_desc = f"Batch {current_batch}/{total_batches} (segments {len(batch)})"
                    with tqdm(total=100, desc=f"   └─ API Call", unit="%", leave=False) as api_pbar:
                        api_pbar.update(10)
                        resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
                        api_pbar.update(40)
                        if resp.status_code == 200:
                            result = resp.json()
                            # try to find text in typical places
                            response_text = None
                            if isinstance(result, dict):
                                # Ollama sometimes returns {'response': '...'}
                                response_text = result.get("response") or result.get("text") or result.get("result") or None
                                # Some Ollama variants include 'output' or list -> join
                                if response_text is None:
                                    # Search nested
                                    for v in result.values():
                                        if isinstance(v, str) and "[" in v and "]" in v:
                                            response_text = v
                                            break
                            elif isinstance(result, str):
                                response_text = result

                            if not response_text:
                                raise ValueError("Ollama response missing 'response' text")

                            # Extract JSON array from response_text
                            json_start = response_text.find("[")
                            json_end = response_text.rfind("]") + 1
                            if json_start >= 0 and json_end > json_start:
                                json_str = response_text[json_start:json_end]
                                batch_translations = json.loads(json_str)
                                translated_segments.extend(batch_translations)
                                api_pbar.update(50)
                                success = True
                            else:
                                # Try parsing as direct JSON
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
                    # Use Google fallback per segment if enabled
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
                                # fallback to original text
                                translated_segments.append({
                                    "start": seg["start"],
                                    "end": seg["end"],
                                    "original": seg["text"],
                                    "translated": seg["text"]
                                })
                    else:
                        # fallback to original
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

        total_elapsed = (datetime.now() - start_time).total_seconds()
        avg_words_per_sec = processed_words / total_elapsed if total_elapsed > 0 else 0
        success_rate = ((total_batches - failed_batches) / total_batches * 100) if total_batches > 0 else 0

        self.log(f"Translation complete: {len(translated_segments)} segments, {processed_words} words")
        # Save translations
        with open(self.translation_file_path, "w", encoding="utf-8") as f:
            json.dump(translated_segments, f, indent=2, ensure_ascii=False)

        # Save text-only version
        text_file = self.dirs["translations"] / f"translations_text_{self.timestamp}.txt"
        with open(text_file, "w", encoding="utf-8") as f:
            for seg in translated_segments:
                f.write(f"[{seg['start']:.2f}s - {seg['end']:.2f}s]\n")
                f.write(f"Original:   {seg['original']}\n")
                f.write(f"Translated: {seg['translated']}\n\n")

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
        except Exception:
            seconds = 0.0

        if seconds < 0:
            seconds = 0.0

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int(round((seconds - math.floor(seconds)) * 1000))

        # handle rounding that produces 1000 milliseconds
        if millis >= 1000:
            millis -= 1000
            secs += 1
            if secs >= 60:
                secs = 0
                minutes += 1
                if minutes >= 60:
                    minutes = 0
                    hours += 1

        sep = ',' if format_type == 'srt' else '.'
        return f"{hours:02}:{minutes:02}:{secs:02}{sep}{millis:03}"

    def create_subtitles(self, translated_segments):
        """Create SRT and VTT subtitle files and a bilingual SRT."""
        self.log("Step 4: Creating subtitle files...")

        self.srt_file_path = self.dirs["subtitles"] / f"subtitles_{self.timestamp}.srt"
        self.vtt_file_path = self.dirs["subtitles"] / f"subtitles_{self.timestamp}.vtt"
        self.bilingual_file_path = self.dirs["subtitles"] / f"subtitles_bilingual_{self.timestamp}.srt"

        # SRT
        with open(self.srt_file_path, "w", encoding="utf-8") as f_srt:
            for i, segment in enumerate(translated_segments, start=1):
                start_srt = self._format_timestamp(segment['start'], 'srt')
                end_srt = self._format_timestamp(segment['end'], 'srt')
                text = segment.get('translated', '')
                f_srt.write(f"{i}\n{start_srt} --> {end_srt}\n{text}\n\n")

        # VTT
        with open(self.vtt_file_path, "w", encoding="utf-8") as f_vtt:
            f_vtt.write("WEBVTT\n\n")
            for i, segment in enumerate(translated_segments, start=1):
                start_vtt = self._format_timestamp(segment['start'], 'vtt')
                end_vtt = self._format_timestamp(segment['end'], 'vtt')
                text = segment.get('translated', '')
                # VTT does not require numeric index but we can keep it for clarity
                f_vtt.write(f"{i}\n{start_vtt} --> {end_vtt}\n{text}\n\n")

        # bilingual SRT (translation first, original italicized)
        with open(self.bilingual_file_path, "w", encoding="utf-8") as f_bi:
            for i, segment in enumerate(translated_segments, start=1):
                start_srt = self._format_timestamp(segment['start'], 'srt')
                end_srt = self._format_timestamp(segment['end'], 'srt')
                original_text = segment.get('original', '')
                translated_text = segment.get('translated', '')
                f_bi.write(f"{i}\n{start_srt} --> {end_srt}\n{translated_text}\n<i>{original_text}</i>\n\n")

        self.log(f"SRT file saved to: {self.srt_file_path}")
        self.log(f"VTT file saved to: {self.vtt_file_path}")
        self.log(f"Bilingual SRT saved to: {self.bilingual_file_path}")

    def generate_report(self, audio_duration, transcript_data, translated_segments, total_start_time):
        """Generate a final summary report"""
        self.log("Step 6: Generating final report...")
        self.report_file_path = self.dirs["root"] / f"translation_report_{self.timestamp}.txt"

        total_elapsed = (datetime.now() - total_start_time).total_seconds()
        rtf = total_elapsed / audio_duration if audio_duration > 0 else 0

        num_segments = len(translated_segments)
        num_words_original = sum(len(s.get('original', '').split()) for s in translated_segments)
        num_words_translated = sum(len(s.get('translated', '').split()) for s in translated_segments)

        report_content = f"""
=================================================
TRANSLATION REPORT
=================================================
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Run ID:    {self.timestamp}
Target:    {self.target_language}

=================================================
SUMMARY
=================================================
Input File:     {self.input_file}
Audio Duration: {audio_duration:.2f}s ({audio_duration/60:.2f} min)
Total Time:     {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)
Processing Speed: {rtf:.2f}x Real-Time Factor (lower is faster)

=================================================
AUDIO (Step 1)
=================================================
Extracted File: {self.audio_file_path.name if self.audio_file_path else 'N/A'}
Format:         16kHz, Mono, PCM 16-bit

=================================================
TRANSCRIPTION (Step 2)
=================================================
Whisper Model:   {WHISPER_MODEL}
Device:          {WHISPER_DEVICE} ({WHISPER_COMPUTE_TYPE})
Detected Lang:   {transcript_data.get('language', 'N/A')} (Prob: {transcript_data.get('language_probability', 0):.1%})
Segments:        {len(transcript_data.get('segments', []))}
Original Words:  {num_words_original}

=================================================
TRANSLATION (Step 3)
=================================================
Ollama Model:     {OLLAMA_MODEL}
Target Language:  {self.target_language}
Segments:         {num_segments}
Translated Words: {num_words_translated}

=================================================
TTS GENERATION (Step 5)
=================================================
TTS Model:       {COQUI_TTS_MODEL}
Speaker Sample:  {COQUI_TTS_SPEAKER} (MUST BE A VALID PATH)
Final TTS Audio: {self.final_tts_file.name if self.final_tts_file else 'N/A'}

=================================================
OUTPUT FILES
=================================================
Log File:       {self.log_file.name}
Extracted Audio: {self.audio_file_path.name if self.audio_file_path else 'N/A'}
Raw Transcript: {self.transcript_file_path.name if self.transcript_file_path else 'N/A'}
Raw Translation: {self.translation_file_path.name if self.translation_file_path else 'N/A'}
SRT Subtitles:  {self.srt_file_path.name if self.srt_file_path else 'N/A'}
VTT Subtitles:  {self.vtt_file_path.name if self.vtt_file_path else 'N/A'}
Bilingual SRT:  {self.bilingual_file_path.name if self.bilingual_file_path else 'N/A'}
Report File:    {self.report_file_path.name}

=================================================
"""

        with open(self.report_file_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        self.log(f"Report saved to: {self.report_file_path}")

    def run(self):
        """Run the full translation pipeline"""
        self.log(f"Starting full translation process for: {self.input_file}")
        total_start_time = datetime.now()

        try:
            # Step 0: Check Ollama if available, but allow fallback behavior
            ollama_ok = self.check_ollama_running() and self.check_ollama_model()
            if not ollama_ok:
                self.log("⚠ Ollama server/model unavailable. Will attempt Google fallback if enabled.")
            else:
                self.log("Ollama server and model confirmed.")

            # Step 1: Extract Audio
            audio_file = self.extract_audio()
            if not audio_file:
                self.log("Error: Audio extraction failed.")
                return

            # Get audio duration for report
            try:
                probe_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                             "-of", "default=noprint_wrappers=1:nokey=1", str(audio_file)]
                audio_duration = float(subprocess.check_output(probe_cmd).decode().strip())
            except Exception:
                audio_duration = 0.0

            # Step 2: Transcribe
            transcript_data = self.transcribe_with_local_whisper(audio_file)
            if not transcript_data or not transcript_data.get('segments'):
                self.log("Error: Transcription failed or produced no segments.")
                return

            # Step 3: Translate
            # If Ollama is not OK, we still call translate_with_ollama because it contains fallback logic.
            translated_segments = self.translate_with_ollama(transcript_data)
            if not translated_segments:
                self.log("Error: Translation failed.")
                return

            # Step 4: Create Subtitles
            self.create_subtitles(translated_segments)
            
            # Step 5: Generate Polish TTS from Segments (uses JSON/segments, not SRT)
            self.synthesize_tts_from_segments(translated_segments, audio_duration)
            self.log(f"🎵 Final Polish audio file: {self.final_tts_file.resolve()}")

            # Step 6: Generate Report
            self.generate_report(audio_duration, transcript_data, translated_segments, total_start_time)

            total_elapsed = (datetime.now() - total_start_time).total_seconds()
            self.log(f"\n{'='*80}")
            self.log(f"🎉 PROCESS FINISHED SUCCESSFULLY in {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)")
            self.log(f"Output files are in: {self.output_dir.resolve()}")
            self.log(f"Report file: {self.report_file_path.resolve()}")
            self.log(f"{'='*80}\n")

        except Exception as e:
            self.log(f"\n{'='*80}")
            self.log(f"❌ FATAL ERROR: An unexpected error occurred: {e}")
            self.log(traceback.format_exc())
            self.log("="*80)

        finally:
            if self.whisper_model:
                self.log("Releasing Whisper model from memory...")
                del self.whisper_model
            if self.tts_engine:
                 self.log("Releasing TTS model from memory...")
                 del self.tts_engine
            if WHISPER_DEVICE == "cuda" or (self.tts_engine and torch.cuda.is_available()):
                torch.cuda.empty_cache()
                self.log("GPU memory cleared.")


if __name__ == "__main__":
    # --- Dependency Checks ---
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, text=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("="*80)
        print("❌ ERROR: FFmpeg is not installed or not in your system's PATH.")
        print("Please install FFmpeg to continue: https://ffmpeg.org/download.html")
        print("="*80)
        exit(1)

    try:
        subprocess.run(["ffprobe", "-version"], check=True, text=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("="*80)
        print("❌ ERROR: ffprobe (part of FFmpeg) is not installed or not in your system's PATH.")
        print("Please install FFmpeg to continue: https://ffmpeg.org/download.html")
        print("="*80)
        exit(1)

    if not Path(INPUT_AUDIO).exists():
        print("="*80)
        print(f"❌ ERROR: Input file not found: {INPUT_AUDIO}")
        print("Please update the INPUT_AUDIO variable at the top of the script.")
        print("="*80)
        exit(1)

    print(f"Starting translation for: {INPUT_AUDIO}")
    print(f"Output will be saved to: {OUTPUT_DIR}")
    print("="*80)
    
    # --- TTS Speaker File Check ---
    # COQUI_TTS_SPEAKER must be a path to a 5-10 second audio file of the original speaker
    # to clone the voice.
    if COQUI_TTS_MODEL.endswith('xtts_v2') and not Path(COQUI_TTS_SPEAKER).is_file():
        print("="*80)
        print(f"🛑 CRITICAL WARNING: Coqui XTTS model requires a speaker sample.")
        print(f"Please change the COQUI_TTS_SPEAKER variable (currently: '{COQUI_TTS_SPEAKER}')")
        print(f"to the **full path** of a 5-10 second WAV/MP3 file of the original speaker's voice.")
        print("The script may fail at Step 5 if this is not corrected.")
        print("="*80)

    translator = LocalAudioTranslator(INPUT_AUDIO, OUTPUT_DIR, LANGUAGE)
    translator.run()