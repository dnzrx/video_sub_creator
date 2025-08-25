import os
import warnings
import subprocess
from typing import Iterator, Optional, NamedTuple
from collections import deque
from enum import Enum

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
MODEL_NAME = ModelSize.SMALL
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
    SUPPORTED_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")
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

    def _extract_audio(self, video_info: VideoInfo) -> Optional[np.ndarray]:
        print(f"Processing: {video_info.filename}")
        print("  - Extracting audio...")
        try:
            cmd = [
                "ffmpeg",
                "-nostdin",
                "-i", video_info.src_path,
                "-f", "s16le",
                "-ac", "1",
                "-ar", "16000",
                "-",
                "-loglevel", "error"
            ]
            result = subprocess.run(cmd, capture_output=True, check=True)
            return np.frombuffer(result.stdout, np.int16).flatten().astype(np.float32) / 32768.0
        except subprocess.CalledProcessError as e:
            print(f"  - Error: FFmpeg failed to extract audio. Details below:")
            print(f"    {e.stderr.decode()}")
            return None

    def _load_model(self):
        if self.model is None:
            model_name_value = MODEL_NAME.value
            print(f"Initializing AI Model: '{model_name_value}'... (This may take a moment on first run)")
            self.model = whisper.load_model(model_name_value)
            print("  - Model loaded successfully.")

    def _process_single_video(self, video_info: VideoInfo):
        audio_data = self._extract_audio(video_info)
        if audio_data is None:
            return

        print("  Generating subtitles...")
        
        warnings.filterwarnings("ignore")
        result = self.model.transcribe(audio_data, **self.TRANSCRIBE_ARGS)
        warnings.filterwarnings("default")

        generator = SubtitleGenerator(result["segments"])

        output_dir = os.path.dirname(video_info.output_path_base)
        os.makedirs(output_dir, exist_ok=True)
        
        generator.write_vtt(video_info.vtt_path)
        generator.write_srt(video_info.srt_path)
        
        print(f"  - Subtitles saved: '{video_info.output_path_base}' (.vtt/.srt)")

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
