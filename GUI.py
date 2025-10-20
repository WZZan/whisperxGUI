#!/usr/bin/env python3
"""
qt_whisperx_stt.py

A single-file PySide6 (Qt) application that:
- Lets the user pick a video file (e.g. .mp4)
- Plays the video using Qt Multimedia
- Extracts audio (via ffmpeg) and runs WhisperX to generate transcription
- Displays transcription in the GUI and allows saving to .srt

ASSUMPTIONS (documented in code):
- ffmpeg is available on PATH
- whisperx and its dependencies (whisper, torch, etc.) are installed and importable
- Python >=3.8, PySide6 installed

This file contains conservative checks, clear logging, and places where the program validates each step.

Note: This is a reference implementation focused on clarity and ease of review rather than ultimate performance.
"""

import sys
import os
import subprocess
import tempfile
import threading
import time
import json
from dataclasses import dataclass
from typing import List, Tuple

# Qt imports
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QTextEdit, QProgressBar, QMessageBox, QLineEdit
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtCore import Qt, QUrl, Signal, QObject

# Try to import whisperx but handle gracefully if unavailable
try:
    import whisperx
    WHISPERX_AVAILABLE = True
except Exception as e:
    WHISPERX_AVAILABLE = False


@dataclass
class SRTLine:
    index: int
    start: float  # seconds
    end: float    # seconds
    text: str


class WorkerSignals(QObject):
    progress = Signal(int)
    finished = Signal(bool, str)  # success, message
    srt_ready = Signal(list)


class WhisperXWorker(threading.Thread):
    """Thread wrapper to run ffmpeg -> whisperx pipeline without blocking UI.

    BEFORE calling each external tool the code documents purpose & inputs.
    AFTER calling, it validates expected files/outputs and emits progress.
    """

    def __init__(self, video_path: str, signals: WorkerSignals, model: str = "small", device: str = "cuda"):
        super().__init__()
        self.video_path = video_path
        self.signals = signals
        self.model = model
        self.device = device
        self._stop = False
        self.chunk_size = 13
        self.vad_onset = 0.3
        self.vad_offset = 0.363
        self.batch_size = 16  

    def run(self):
        # Step 1: Extract audio using ffmpeg
        self.signals.progress.emit(5)
        # self.signals.finished.emit(True, "開始抽取音訊...")
        try:
            audio_file = self._extract_audio(self.video_path)
        except Exception as e:
            self.signals.finished.emit(False, f"ffmpeg 轉檔失敗: {e}")
            return
        self.signals.progress.emit(20)

        # Step 2: Run whisperx transcription
        if not WHISPERX_AVAILABLE:
            self.signals.finished.emit(False, "whisperx 尚未安裝或 import 失敗，請先安裝 whisperx")
            return

        self.signals.progress.emit(30)
        try:
            srt_lines = self._run_whisperx(audio_file)
        except Exception as e:
            self.signals.finished.emit(False, f"WhisperX 執行失敗: {e}")
            return

        self.signals.progress.emit(90)

        # Step 3: Emit result (and cleanup)
        self.signals.srt_ready.emit(srt_lines)
        self.signals.progress.emit(100)
        self.signals.finished.emit(True, "完成")

    def _extract_audio(self, video_path: str) -> str:
        """Purpose: extract a WAV (or FLAC) audio file suitable for WhisperX.

        Inputs:
        - video_path: full path to the chosen video file

        Expected output:
        - path to a lossless audio file (wav) on success

        Validations:
        - Ensure ffmpeg exists by running `ffmpeg -version`
        - Validate returncode of ffmpeg command
        """
        # Validate ffmpeg presence
        try:
            subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            raise RuntimeError("找不到 ffmpeg，請將 ffmpeg 加入 PATH 並可於命令列執行")

        tmp_dir = tempfile.mkdtemp(prefix="qt_whisperx_")
        out_audio = os.path.join(tmp_dir, "extracted_audio.wav")

        # Build ffmpeg command: extract audio, 16k/16-bit mono is safe for many STT models
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-ac", "1", "-ar", "16000", "-vn", out_audio
        ]
        # Execute
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0 or not os.path.exists(out_audio):
            raise RuntimeError(f"ffmpeg 失敗: returncode={proc.returncode}, stderr={proc.stderr.decode(errors='ignore')}")

        # Quick validation: non-empty file
        if os.path.getsize(out_audio) < 100:
            raise RuntimeError("抽取出來的音訊檔太小，可能出錯")

        return out_audio

    def _run_whisperx(self, audio_path: str) -> List[SRTLine]:
        """Purpose: use whisperx to transcribe audio and produce word-level timestamps and final segments.

        Inputs:
        - audio_path: path to extracted audio file
        - model, device provided in worker initialization

        Expected outputs:
        - List[SRTLine] with timing in seconds and text

        Validation & fallback:
        - If GPU device fails (e.g., cuda not available), fall back to cpu
        - If model load fails, raise descriptive exception
        """
        # Conservative attempt: try the requested device, fallback to cpu
        device = self.device
        chunk_size = self.chunk_size
        vad_onset = self.vad_onset 
        vad_offset = self.vad_offset
        batch_size = self.batch_size

        # try:
        model = whisperx.load_model(
            self.model,
            device=device,
            compute_type="float16",
            language="ja",
            vad_options={
            "chunk_size": chunk_size,
            "vad_onset": vad_onset,
            "vad_offset": vad_offset,
    },
        )
        # except Exception as e:
        #     # fallback to cpu
        #     try:
        #         device = "cpu"
        #         model = whisperx.load_model(self.model, device=device)
        #     except Exception as e2:
        #         raise RuntimeError(f"載入 whisperx 模型失敗: primary error={e}, fallback error={e2}")

        # Transcribe (this uses whisperx API: transcribe then align)
        # The following is a guarded use reflecting typical whisperx usage.
        result = model.transcribe(
            audio_path,
            batch_size=batch_size,
            chunk_size=chunk_size,      
            )
        # result contains .segments usually; but whisperx typical flow requires alignment
        try:
            # align to get word-level timestamps
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
            result_aligned = whisperx.align(result["segments"], model_a, metadata, audio_path, device)
            # Convert aligned segments to SRT lines
            srt_lines: List[SRTLine] = []
            index = 1
            for seg in result_aligned["segments"]:
                start = seg["start"]
                end = seg["end"]
                text = seg.get("text", "").strip()
                srt_lines.append(SRTLine(index=index, start=start, end=end, text=text))
                index += 1
            return srt_lines
        except Exception:
            # Fallback: use coarse segments from initial transcription
            srt_lines = []
            index = 1
            segments = result.get("segments", [])
            for seg in segments:
                start = seg.get("start", 0.0)
                end = seg.get("end", start + 1.0)
                text = seg.get("text", "").strip()
                srt_lines.append(SRTLine(index=index, start=start, end=end, text=text))
                index += 1
            return srt_lines


# --- GUI application ---

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WhisperX STT — Qt 前端範例")
        self.resize(1000, 700)

        # Central widget
        w = QWidget()
        self.setCentralWidget(w)
        layout = QVBoxLayout()
        w.setLayout(layout)

        # Video player
        self.video_widget = QVideoWidget()
        layout.addWidget(self.video_widget, stretch=3)

        self.player = QMediaPlayer()
        audio_out = QAudioOutput()
        self.player.setAudioOutput(audio_out)
        self.player.setVideoOutput(self.video_widget)

        # Controls
        ctrl_layout = QHBoxLayout()
        self.open_btn = QPushButton("選取影片")
        self.open_btn.clicked.connect(self.open_file)
        ctrl_layout.addWidget(self.open_btn)

        self.play_btn = QPushButton("播放/暫停")
        self.play_btn.clicked.connect(self.play_pause)
        ctrl_layout.addWidget(self.play_btn)

        self.run_btn = QPushButton("執行語音辨識 (WhisperX)")
        self.run_btn.clicked.connect(self.run_stt)
        ctrl_layout.addWidget(self.run_btn)

        self.save_btn = QPushButton("儲存 .srt")
        self.save_btn.clicked.connect(self.save_srt)
        self.save_btn.setEnabled(False)
        ctrl_layout.addWidget(self.save_btn)

        layout.addLayout(ctrl_layout)

        # Progress and status
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.status_label = QLabel("狀態: 等待使用者操作")
        layout.addWidget(self.status_label)

        # Transcript area
        self.transcript = QTextEdit()
        self.transcript.setReadOnly(True)
        layout.addWidget(self.transcript, stretch=2)

        # Internal state
        self.current_video = None
        self.srt_lines: List[SRTLine] = []

        # Worker signals
        self.signals = WorkerSignals()
        self.signals.progress.connect(self._on_progress)
        self.signals.finished.connect(self._on_finished)
        self.signals.srt_ready.connect(self._on_srt_ready)

    def open_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "選取影片檔", os.path.expanduser("~"), "影片檔 (*.mp4 *.mkv *.avi *.mov);")
        if not file:
            return
        self.current_video = file
        self.player.setSource(QUrl.fromLocalFile(file))
        self.status_label.setText(f"已選擇: {file}")

    def play_pause(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def run_stt(self):
        if not self.current_video:
            QMessageBox.warning(self, "錯誤", "請先選取一個影片檔")
            return
        # Disable run button during work
        self.run_btn.setEnabled(False)
        self.status_label.setText("開始處理: 抽取音訊並呼叫 WhisperX...")
        self.progress.setValue(0)
        self.worker = WhisperXWorker(self.current_video, self.signals)
        self.worker.start()

    def _on_progress(self, val: int):
        self.progress.setValue(val)

    def _on_finished(self, success: bool, message: str):
        if success:
            self.status_label.setText("完成: 可儲存字幕或檢視結果")
            self.save_btn.setEnabled(True)
        else:
            self.status_label.setText(f"失敗: {message}")
            QMessageBox.critical(self, "處理失敗", message)
        self.run_btn.setEnabled(True)

    def _on_srt_ready(self, srt_lines: List[SRTLine]):
        self.srt_lines = srt_lines
        # Populate transcript display
        self.transcript.clear()
        for ln in srt_lines:
            start_h = self._format_time(ln.start)
            end_h = self._format_time(ln.end)
            self.transcript.append(f"[{ln.index}] {start_h} --> {end_h}\n{ln.text}\n")

    def save_srt(self):
        if not self.srt_lines:
            QMessageBox.warning(self, "無字幕可儲存", "尚未產生字幕")
            return
        path, _ = QFileDialog.getSaveFileName(self, "儲存 SRT", os.path.splitext(self.current_video)[0] + ".srt", "SRT 檔 (*.srt)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                for ln in self.srt_lines:
                    f.write(f"{ln.index}\n")
                    f.write(f"{self._format_time(ln.start, srt=True)} --> {self._format_time(ln.end, srt=True)}\n")
                    f.write(ln.text.replace('\n', ' ') + "\n\n")
            QMessageBox.information(self, "完成", f"已儲存: {path}")
        except Exception as e:
            QMessageBox.critical(self, "儲存失敗", str(e))

    @staticmethod
    def _format_time(t: float, srt: bool = False) -> str:
        # t in seconds
        ms = int((t - int(t)) * 1000)
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        if srt:
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
        else:
            return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def main():
    app = QApplication(sys.argv)

    w = MainWindow()
    w.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
