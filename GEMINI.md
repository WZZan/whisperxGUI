# Project: whisperXGUI

## Project Overview

This project is a graphical user interface (GUI) application for the `whisperx` speech recognition library. It is built using Python with the PySide6 (Qt) framework. The application allows users to select a video file, and it will then extract the audio, transcribe it using `whisperx`, and display the resulting transcription. The transcription can also be saved as a `.srt` subtitle file.

The application is designed to be a single-file, standalone tool. It uses `ffmpeg` for audio extraction, which must be installed separately on the system.

## Key Files

*   `GUI.py`: This is the main and only source file for the application. It contains all the logic for the GUI, video playback, and the worker thread that handles the `ffmpeg` and `whisperx` processing.
*   `pyproject.toml`: This file defines the project metadata and dependencies. It specifies the required Python version (`>=3.10`) and the necessary Python packages (`pyside6` and `whisperx`).
*   `README.md`: The project's README file. (Currently empty)
*   `.python-version`: Specifies that the project uses python version 3.12.4.

## Dependencies

### Python Packages

The Python dependencies are listed in `pyproject.toml`:

*   `pyside6>=6.10.0`: For the graphical user interface.
*   `whisperx>=3.7.4`: For the speech-to-text transcription.

### External Dependencies

*   `ffmpeg`: This is required for extracting audio from video files. It must be installed on the system and accessible from the command line (i.e., in the system's PATH).

## How to Run the Application

To run the application, execute the `GUI.py` script from your terminal:

```sh
python GUI.py
```

Make sure you have installed the required Python packages and `ffmpeg` before running the application.

## Development Conventions

*   **Single-File Application:** All the application code is contained within `GUI.py`.
*   **Threading:** A `WhisperXWorker` class (which inherits from `threading.Thread`) is used to run the time-consuming audio extraction and transcription processes in a separate thread. This prevents the GUI from freezing during these operations.
*   **Dependency Checks:** The application includes checks to ensure that `ffmpeg` is available and that the `whisperx` library can be imported.
*   **Error Handling:** The application provides feedback to the user in case of errors, such as a missing video file or a failure in the transcription process.
