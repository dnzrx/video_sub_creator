# Automatic Video Subtitle Generator

This project uses OpenAI's Whisper model to automatically generate subtitle files (`.srt` and `.vtt`) for any video. It processes all videos in a designated `video_src` folder and saves the subtitle files to a `result` directory.

## Features

-   **Automatic Transcription:** Uses OpenAI's Whisper `small` model to generate accurate subtitles.
-   **Batch Processing:** Processes all videos in the `video_src` directory.
-   **Dual Format:** Creates both `.srt` and `.vtt` subtitle files.
-   **Bundled FFmpeg:** Includes its own version of FFmpeg, so no separate installation is required.
-   **Auto-Language Detection:** Automatically detects the language of the audio.

## Installation

To get started, you'll need Python 3.9 or newer.

1.  **Create a Virtual Environment:**
    ```bash
    python -m venv .venv
    .venv\Scripts\activate
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    pip install -e .
    ```

## Usage

1.  **Add Videos:** Place all the videos you want to transcribe into the `video_src` folder.

2.  **Run the Script:** Open your terminal and run the following command:
    ```bash
    video_sub_creator
    ```

The script will automatically detect your hardware and use it for transcription. The generated `.srt` and `.vtt` subtitle files will be saved to the `result` directory.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.