# Automatic Video Subtitle Generator

This project uses OpenAI's Whisper model to automatically generate subtitle files (`.srt` and `.vtt`) for videos. It recursively scans a `video_src` folder, processes any videos it finds, and saves the subtitle files to a `result` directory while preserving the original folder structure.

## Features

-   **Automatic Transcription:** Uses OpenAI's Whisper model to generate accurate subtitles.
-   **Recursive Processing:** Scans all subdirectories in the `video_src` folder for videos.
-   **Mirrored Output Structure:** Replicates the source directory structure in the `result` folder.
-   **Configurable Model:** Easily change the Whisper model size (`tiny`, `base`, `small`, `medium`, `large`) in the `main.py` script.
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

1.  **Add Videos:** Place your video files and folders into the `video_src` directory. The script will find videos in any subdirectories.

2.  **(Optional) Configure Parameters:** Open `video_sub_creator/main.py` and modify the values in the configuration section at the top of the file.

    For example:
    ```python
    # --- Configurable parameters ---
    VIDEO_SRC_DIR = "my_videos"
    OUTPUT_DIR = "subtitles"

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
    MODEL_NAME = ModelSize.LARGE
    # --- End of Configurable parameters ---
    ```

3.  **Run the Script:** Open your terminal and run the following command:
    ```bash
    video_sub_creator
    ```

The script will automatically detect your hardware and use it for transcription. The generated `.srt` and `.vtt` subtitle files will be saved to the `result` directory, mirroring the structure of `video_src`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
