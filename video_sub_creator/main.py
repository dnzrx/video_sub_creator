import os
import warnings
import subprocess
from typing import Iterator, Optional
import concurrent.futures

import whisper
import numpy as np
import static_ffmpeg

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
    VIDEO_SRC_DIR = "video_src"
    OUTPUT_DIR = "result"
    MODEL_NAME = "small"
    SUPPORTED_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")
    TRANSCRIBE_ARGS = {
        "task": "transcribe",
        "language": None,
        "verbose": False,
    }

    def __init__(self):
        self.model = None

    @staticmethod
    def _get_filename(path: str) -> str:
        return os.path.splitext(os.path.basename(path))[0]

    def _get_video_files(self) -> Iterator[str]:
        if not os.path.isdir(self.VIDEO_SRC_DIR):
            print(f"Error: Source directory not found: '{self.VIDEO_SRC_DIR}'")
            return
        for f in os.listdir(self.VIDEO_SRC_DIR):
            if f.endswith(self.SUPPORTED_EXTENSIONS):
                yield os.path.join(self.VIDEO_SRC_DIR, f)

    def _extract_audio(self, video_path: str) -> Optional[np.ndarray]:
        video_filename = self._get_filename(video_path)
        print(f"Processing: {video_filename}")
        print("  - Extracting audio...")
        try:
            cmd = [
                "ffmpeg",
                "-nostdin",
                "-i", video_path,
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
            print(f"Initializing AI Model: '{self.MODEL_NAME}'... (This may take a moment on first run)")
            self.model = whisper.load_model(self.MODEL_NAME)
            print("  - Model loaded successfully.")

    def _process_single_video(self, video_path: str):
        audio_data = self._extract_audio(video_path)
        if audio_data is None:
            return

        video_filename = self._get_filename(video_path)
        print("  Generating subtitles...")
        
        warnings.filterwarnings("ignore")
        result = self.model.transcribe(audio_data, **self.TRANSCRIBE_ARGS)
        warnings.filterwarnings("default")

        generator = SubtitleGenerator(result["segments"])
        
        vtt_path = os.path.join(self.OUTPUT_DIR, f"{video_filename}.vtt")
        generator.write_vtt(vtt_path)

        srt_path = os.path.join(self.OUTPUT_DIR, f"{video_filename}.srt")
        generator.write_srt(srt_path)
        
        print(f"  - Subtitles saved: '{os.path.join(self.OUTPUT_DIR, video_filename)}' (.vtt/.srt)")

    def run(self):
        static_ffmpeg.add_paths()
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        
        video_files = list(self._get_video_files())
        if not video_files:
            print(f"No videos found in '{self.VIDEO_SRC_DIR}'. Please add video files to begin.")
            return
        
        self._load_model()

        video_count = len(video_files)
        print(f"\nFound {video_count} video(s) to process. Starting job...\n")
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self._process_single_video, path): path for path in video_files}
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    video_path = futures[future]
                    print(f"An unexpected error occurred while processing '{self._get_filename(video_path)}': {e}")
        
        print("\nJob finished. All videos processed successfully.")

def main():
    processor = VideoProcessor()
    processor.run()

if __name__ == '__main__':
    main()
