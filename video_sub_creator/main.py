import os
import warnings
import subprocess
from typing import Iterator, Optional, NamedTuple
from collections import deque
from enum import Enum
import tempfile

from pydub import AudioSegment
from pydub.silence import detect_nonsilent

import whisper
import numpy as np
import static_ffmpeg


# --- Configurable parameters ---
VIDEO_SRC_DIR = "video_src"
OUTPUT_DIR = "result"

class ModelSize(Enum):
    """
    Available Whisper model sizes.
    See: https://github.com/openai/whisper#available-models-and-languages
    """
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

# Select the model size to use for transcription
# Options: ModelSize.TINY, ModelSize.BASE, ModelSize.SMALL, ModelSize.MEDIUM, ModelSize.LARGE
MODEL_NAME = ModelSize.MEDIUM

# Silence splitting parameters
MIN_SILENCE_LEN = 1500  # ms
SILENCE_THRESH = -40    # dBFS
# --- End of Configurable parameters ---


class VideoInfo(NamedTuple):
    src_path: str
    output_path_base: str

    @property
    def vtt_path(self) -> str:
        return f"{self.output_path_base}.vtt"

    @property
    def srt_path(self) -> str:
        return f"{self.output_path_base}.srt"

    @property
    def filename(self) -> str:
        return os.path.splitext(os.path.basename(self.src_path))[0]


class SubtitleGenerator:
    def __init__(self, transcript: Iterator[dict]):
        self.transcript = transcript

    def _format_timestamp(self, seconds: float, decimal_marker: str = '.') -> str:
        assert seconds >= 0, "non-negative timestamp expected"
        milliseconds = round(seconds * 1000.0)
        hours = milliseconds // 3_600_000
        milliseconds -= hours * 3_600_000
        minutes = milliseconds // 60_000
        milliseconds -= minutes * 60_000
        seconds = milliseconds // 1_000
        milliseconds -= seconds * 1_000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"

    def write_vtt(self, file_path: str):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for segment in self.transcript:
                start = self._format_timestamp(segment['start'])
                end = self._format_timestamp(segment['end'])
                text = segment['text'].strip().replace('-->', '->')
                f.write(f"{start} --> {end}\n{text}\n\n")

    def write_srt(self, file_path: str):
        with open(file_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(self.transcript, start=1):
                start = self._format_timestamp(segment['start'], decimal_marker=',')
                end = self._format_timestamp(segment['end'], decimal_marker=',')
                text = segment['text'].strip().replace('-->', '->')
                f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

class VideoProcessor:
    SUPPORTED_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", "m4a")
    TRANSCRIBE_ARGS = {
        "task": "transcribe",
        "language": None,
        "verbose": False,
    }

    def __init__(self):
        self.model = None

    def _get_video_files(self) -> Iterator[str]:
        if not os.path.isdir(VIDEO_SRC_DIR):
            print(f"Error: Source directory not found: '{VIDEO_SRC_DIR}'")
            return
        for root, _, files in os.walk(VIDEO_SRC_DIR):
            for f in files:
                if f.endswith(self.SUPPORTED_EXTENSIONS):
                    yield os.path.join(root, f)

    def _extract_audio(self, video_info: VideoInfo) -> Optional[str]:
        print(f"Processing: {video_info.filename}")
        print("  - Extracting audio...")
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            audio_path = tmp_file.name

        try:
            cmd = [
                "ffmpeg",
                "-nostdin",
                "-i", video_info.src_path,
                "-vn", # No video
                "-acodec", "pcm_s16le", # WAV format
                "-ac", "1", # Mono
                "-ar", "16000", # 16kHz sample rate
                "-y", # Overwrite output file
                audio_path,
                "-loglevel", "error"
            ]
            subprocess.run(cmd, check=True)
            return audio_path
        except subprocess.CalledProcessError as e:
            print(f"  - Error: FFmpeg failed to extract audio. Details below:")
            # print(f"    {e.stderr.decode()}") This is not available with check=True and no capture_output
            return None

    def _convert_segment_to_whisper_format(self, segment: AudioSegment) -> np.ndarray:
        samples = np.array(segment.get_array_of_samples())
        return samples.astype(np.float32) / 32768.0

    def _load_model(self):
        if self.model is None:
            model_name_value = MODEL_NAME.value
            print(f"Initializing AI Model: '{model_name_value}'... (This may take a moment on first run)")
            self.model = whisper.load_model(model_name_value)
            print("  - Model loaded successfully.")

    def _generate_segments(self, audio_path: str) -> list[dict]:
        """
        Loads audio, detects non-silent parts, transcribes them, and returns subtitle segments.
        """
        print("  - Loading audio for segmentation...")
        with open(audio_path, "rb") as f:
            audio = AudioSegment.from_file(f, format="wav")
        
        print("  - Detecting non-silent parts...")
        nonsilent_ranges = detect_nonsilent(audio, min_silence_len=MIN_SILENCE_LEN, silence_thresh=SILENCE_THRESH)

        if not nonsilent_ranges:
            print("  - No non-silent audio detected. Transcribing the whole audio.")
            audio_data = self._convert_segment_to_whisper_format(audio)
            
            if audio_data.size == 0:
                print("  - Audio is empty after conversion. Skipping.")
                return []

            warnings.filterwarnings("ignore")
            result = self.model.transcribe(audio_data, **self.TRANSCRIBE_ARGS)
            warnings.filterwarnings("default")
            return result.get("segments", [])

        all_segments = []
        print(f"  - Transcribing {len(nonsilent_ranges)} audio segments...")
        for start_ms, end_ms in nonsilent_ranges:
            chunk = audio[start_ms:end_ms]
            audio_data = self._convert_segment_to_whisper_format(chunk)

            if audio_data.size == 0:
                continue

            warnings.filterwarnings("ignore")
            result = self.model.transcribe(audio_data, **self.TRANSCRIBE_ARGS)
            warnings.filterwarnings("default")

            if result and "segments" in result and result["segments"]:
                for seg in result["segments"]:
                    seg["start"] += start_ms / 1000.0
                    seg["end"] += start_ms / 1000.0
                all_segments.extend(result["segments"])
        
        return all_segments

    def _process_single_video(self, video_info: VideoInfo):
        audio_path = self._extract_audio(video_info)
        if audio_path is None:
            return
        
        try:
            all_segments = self._generate_segments(audio_path)
            
            if not all_segments:
                print("  - Transcription resulted in no segments.")
                return

            generator = SubtitleGenerator(all_segments)

            output_dir = os.path.dirname(video_info.output_path_base)
            os.makedirs(output_dir, exist_ok=True)
            
            generator.write_vtt(video_info.vtt_path)
            generator.write_srt(video_info.srt_path)
            
            print(f"  - Subtitles saved: '{video_info.output_path_base}' (.vtt/.srt)")
        finally:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)

    def run(self):
        static_ffmpeg.add_paths()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        video_queue = deque()
        for path in self._get_video_files():
            relative_path = os.path.relpath(path, VIDEO_SRC_DIR)
            output_base = os.path.join(OUTPUT_DIR, os.path.splitext(relative_path)[0])
            video_queue.append(VideoInfo(src_path=path, output_path_base=output_base))

        if not video_queue:
            print(f"No videos found in '{VIDEO_SRC_DIR}'. Please add video files to begin.")
            return
        
        self._load_model()

        video_count = len(video_queue)
        print(f"\nFound {video_count} video(s) to process. Starting job...\n")
        
        failed_videos = []
        while video_queue:
            video_info = video_queue.popleft()
            try:
                self._process_single_video(video_info)
            except Exception as e:
                print(f"An unexpected error occurred while processing '{video_info.filename}': {e}")
                failed_videos.append(video_info.filename)
        
        print("\nJob finished.")
        if failed_videos:
            print("\nThe following videos failed to process:")
            for video_name in failed_videos:
                print(f"  - {video_name}")
        else:
            print("All videos processed successfully.")

def main():
    processor = VideoProcessor()
    processor.run()

if __name__ == '__main__':
    main()
