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
from pathlib import Path

# Qt imports
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QTextEdit, QProgressBar, QMessageBox, QLineEdit, QComboBox, QToolButton, QSlider
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtCore import Qt, QUrl, Signal, QObject
from PySide6.QtGui import QIcon


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

    def __init__(self, video_path: str, signals: WorkerSignals, model: str = "medium", device: str = "cuda", language: str = "ja", chunk_size: int = 13, vad_onset: float = 0.3, vad_offset: float = 0.363, batch_size: int = 16, no_speech_threshold: float = 0.5, compute_type: str = "float16"):
        super().__init__()
        self.video_path = video_path
        self.signals = signals
        self.model = model
        self.device = device
        self.language = language
        self._stop = False
        self.chunk_size = chunk_size
        self.vad_onset = vad_onset
        self.vad_offset = vad_offset
        self.batch_size = batch_size
        self.no_speech_threshold = no_speech_threshold
        self.compute_type = compute_type

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
        compute_type = self.compute_type

        try:
            model = whisperx.load_model(
                self.model,
                device=device,
                compute_type=compute_type,
                language=self.language,
                vad_options={
                    "chunk_size": chunk_size,
                    "vad_onset": vad_onset,
                    "vad_offset": vad_offset,
                },
                asr_options={
                    "no_speech_threshold": self.no_speech_threshold,
                    },
            )
        except Exception as e:
            # fallback to cpu
            try:
                device = "cpu"
                model = whisperx.load_model(self.model, device=device)
            except Exception as e2:
                raise RuntimeError(f"載入 whisperx 模型失敗: primary error={e}, fallback error={e2}")

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
    CONFIG_FILE = "whisperx_config.json"

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

        # Video progress slider
        self.video_slider = QSlider(Qt.Horizontal)
        self.video_slider.setSliderPosition(0)
        self.video_slider.sliderMoved.connect(self.seek_video)
        layout.addWidget(self.video_slider)

        self.player = QMediaPlayer()
        audio_out = QAudioOutput()
        self.player.setAudioOutput(audio_out)
        self.player.setVideoOutput(self.video_widget)
        self.player.positionChanged.connect(self.update_slider_position)
        self.player.durationChanged.connect(self.update_slider_range)

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

        # --- Settings ---
        self.settings_widget = QWidget()
        settings_layout = QVBoxLayout()
        self.settings_widget.setLayout(settings_layout)
        layout.addWidget(self.settings_widget)

        # --- Basic Settings ---
        basic_settings_layout = QHBoxLayout()

        # Model selection
        model_group = QWidget()
        model_layout = QVBoxLayout()
        model_group.setLayout(model_layout)
        model_layout.addWidget(QLabel("WhisperX Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large"])
        self.model_combo.setCurrentText("medium")
        model_layout.addWidget(self.model_combo)
        basic_settings_layout.addWidget(model_group)

        # Device selection
        device_group = QWidget()
        device_layout = QVBoxLayout()
        device_group.setLayout(device_layout)
        device_layout.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda", "cpu"])
        self.device_combo.setCurrentText("cuda")
        device_layout.addWidget(self.device_combo)
        basic_settings_layout.addWidget(device_group)

        # Language selection
        language_group = QWidget()
        language_layout = QVBoxLayout()
        language_group.setLayout(language_layout)
        language_layout.addWidget(QLabel("Language:"))
        self.language_input = QLineEdit("ja")
        language_layout.addWidget(self.language_input)
        basic_settings_layout.addWidget(language_group)

        # Batch size
        batch_size_group = QWidget()
        batch_size_layout = QVBoxLayout()
        batch_size_group.setLayout(batch_size_layout)
        batch_size_layout.addWidget(QLabel("Batch Size:"))
        self.batch_size_input = QLineEdit("16")
        batch_size_layout.addWidget(self.batch_size_input)
        basic_settings_layout.addWidget(batch_size_group)

        # Compute Type
        compute_type_group = QWidget()
        compute_type_layout = QVBoxLayout()
        compute_type_group.setLayout(compute_type_layout)
        compute_type_layout.addWidget(QLabel("Compute Type:"))
        self.compute_type_combo = QComboBox()
        self.compute_type_combo.addItems(["int8", "float16"])
        self.compute_type_combo.setCurrentText("float16")
        compute_type_layout.addWidget(self.compute_type_combo)
        basic_settings_layout.addWidget(compute_type_group)

        settings_layout.addLayout(basic_settings_layout)

        # --- Advanced Settings Toggle ---
        self.adv_settings_toggle_btn = QToolButton()
        self.adv_settings_toggle_btn.setArrowType(Qt.RightArrow)
        self.adv_settings_toggle_btn.setCheckable(True)
        self.adv_settings_toggle_btn.setChecked(False)
        self.adv_settings_toggle_btn.clicked.connect(self.toggle_advanced_settings)
        
        adv_toggle_layout = QHBoxLayout()
        adv_toggle_layout.addWidget(self.adv_settings_toggle_btn)
        adv_toggle_layout.addWidget(QLabel("Advanced Settings"))
        adv_toggle_layout.addStretch()
        settings_layout.addLayout(adv_toggle_layout)

        # --- Advanced Settings ---
        self.adv_settings_widget = QWidget()
        adv_settings_layout = QHBoxLayout()
        self.adv_settings_widget.setLayout(adv_settings_layout)
        settings_layout.addWidget(self.adv_settings_widget)
        self.adv_settings_widget.setVisible(False)

        # Chunk size
        chunk_size_group = QWidget()
        chunk_size_layout = QVBoxLayout()
        chunk_size_group.setLayout(chunk_size_layout)
        chunk_size_layout.addWidget(QLabel("Chunk Size:"))
        self.chunk_size_input = QLineEdit("13")
        chunk_size_layout.addWidget(self.chunk_size_input)
        adv_settings_layout.addWidget(chunk_size_group)

        # VAD Onset
        vad_onset_group = QWidget()
        vad_onset_layout = QVBoxLayout()
        vad_onset_group.setLayout(vad_onset_layout)
        vad_onset_layout.addWidget(QLabel("VAD Onset:"))
        self.vad_onset_input = QLineEdit("0.3")
        vad_onset_layout.addWidget(self.vad_onset_input)
        adv_settings_layout.addWidget(vad_onset_group)

        # VAD Offset
        vad_offset_group = QWidget()
        vad_offset_layout = QVBoxLayout()
        vad_offset_group.setLayout(vad_offset_layout)
        vad_offset_layout.addWidget(QLabel("VAD Offset:"))
        self.vad_offset_input = QLineEdit("0.363")
        vad_offset_layout.addWidget(self.vad_offset_input)
        adv_settings_layout.addWidget(vad_offset_group)

        # No Speech Threshold
        no_speech_threshold_group = QWidget()
        no_speech_threshold_layout = QVBoxLayout()
        no_speech_threshold_group.setLayout(no_speech_threshold_layout)
        no_speech_threshold_layout.addWidget(QLabel("No Speech Threshold:"))
        self.no_speech_threshold_input = QLineEdit("0.5")
        no_speech_threshold_layout.addWidget(self.no_speech_threshold_input)
        adv_settings_layout.addWidget(no_speech_threshold_group)

        # Internal state
        self.current_video = None
        self.srt_lines: List[SRTLine] = []

        # Worker signals
        self.signals = WorkerSignals()
        self.signals.progress.connect(self._on_progress)
        self.signals.finished.connect(self._on_finished)
        self.signals.srt_ready.connect(self._on_srt_ready)

        # Load saved configuration
        self.load_config()

    def closeEvent(self, event):
        """Save configuration before closing the application."""
        self.save_config()
        event.accept()

    def toggle_advanced_settings(self):
        """Toggle the visibility of advanced settings."""
        is_checked = self.adv_settings_toggle_btn.isChecked()
        self.adv_settings_widget.setVisible(is_checked)
        self.adv_settings_toggle_btn.setArrowType(Qt.DownArrow if is_checked else Qt.RightArrow)

    def seek_video(self, position: int):
        """Seek video to the specified position when slider is moved."""
        self.player.setPosition(position)

    def update_slider_position(self, position: int):
        """Update slider position based on current video playback position."""
        self.video_slider.blockSignals(True)
        self.video_slider.setValue(position)
        self.video_slider.blockSignals(False)

    def update_slider_range(self, duration: int):
        """Update slider range based on video duration."""
        self.video_slider.setRange(0, duration)

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

        # Get settings from UI
        model = self.model_combo.currentText()
        device = self.device_combo.currentText()
        language = self.language_input.text()
        compute_type = self.compute_type_combo.currentText()
        try:
            chunk_size = int(self.chunk_size_input.text())
            vad_onset = float(self.vad_onset_input.text())
            vad_offset = float(self.vad_offset_input.text())
            batch_size = int(self.batch_size_input.text())
            no_speech_threshold = float(self.no_speech_threshold_input.text())
        except ValueError:
            QMessageBox.warning(self, "錯誤", "進階設定中的參數必須是數字")
            return

        # Disable run button during work
        self.run_btn.setEnabled(False)
        self.status_label.setText("開始處理: 抽取音訊並呼叫 WhisperX...")
        self.progress.setValue(0)
        self.worker = WhisperXWorker(
            self.current_video, self.signals, model=model, device=device, language=language,
            chunk_size=chunk_size, vad_onset=vad_onset, vad_offset=vad_offset, batch_size=batch_size,
            no_speech_threshold=no_speech_threshold, compute_type=compute_type 
        )
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

    def save_config(self):
        """Save all settings to JSON configuration file."""
        config = {
            "model": self.model_combo.currentText(),
            "device": self.device_combo.currentText(),
            "language": self.language_input.text(),
            "compute_type": self.compute_type_combo.currentText(),
            "batch_size": self.batch_size_input.text(),
            "chunk_size": self.chunk_size_input.text(),
            "vad_onset": self.vad_onset_input.text(),
            "vad_offset": self.vad_offset_input.text(),
            "no_speech_threshold": self.no_speech_threshold_input.text(),
        }
        try:
            with open(self.CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"儲存配置文件失敗: {e}")

    def load_config(self):
        """Load settings from JSON configuration file."""
        if not os.path.exists(self.CONFIG_FILE):
            return
        try:
            with open(self.CONFIG_FILE, "r", encoding="utf-8") as f:
                config = json.load(f)
            
            # Apply loaded settings to UI
            if "model" in config:
                self.model_combo.setCurrentText(config["model"])
            if "device" in config:
                self.device_combo.setCurrentText(config["device"])
            if "language" in config:
                self.language_input.setText(config["language"])
            if "compute_type" in config:
                self.compute_type_combo.setCurrentText(config["compute_type"])
            if "batch_size" in config:
                self.batch_size_input.setText(config["batch_size"])
            if "chunk_size" in config:
                self.chunk_size_input.setText(config["chunk_size"])
            if "vad_onset" in config:
                self.vad_onset_input.setText(config["vad_onset"])
            if "vad_offset" in config:
                self.vad_offset_input.setText(config["vad_offset"])
            if "no_speech_threshold" in config:
                self.no_speech_threshold_input.setText(config["no_speech_threshold"])
        except Exception as e:
            print(f"加載配置文件失敗: {e}")


def main():
    app = QApplication(sys.argv)

    w = MainWindow()
    w.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

