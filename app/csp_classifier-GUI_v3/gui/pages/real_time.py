import os
import pickle
import traceback
import numpy as np
import joblib
import math
from threading import Thread
from collections import deque

from PySide6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QScrollArea, QCheckBox, QSpinBox
from PySide6.QtCore import Qt, Signal, QObject

import mne
from scipy.signal import welch
import sklearn
import pyriemann
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

from gui.colors import colors
from gui.fonts import fonts
from gui.components.back_button import BackButton
from gui.components.title_label import TitleLabel
from config.config import Configurations
from src.bci_integration.GtecNautilusProInterface import GtecNautilusProInterface
from src.bci_integration.MockBCIInterface import MockBCIInterface
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class ThreadSafePredictionSignaler(QObject):
    update_signal = Signal(str, str, float, float)

    def __init__(self, update_callback):
        super().__init__()
        self.update_callback = update_callback
        self.update_signal.connect(self.update_callback)

    def __call__(self, label, probs_text, prob_value, raw_value):
        self.update_signal.emit(label, probs_text, prob_value, raw_value)


class DummyPromptViewer:
    def __init__(self):
        self.closed = False
    def change_prompt(self, prompt):
        pass


class DummyTimelineSignaler:
    def __call__(self, val):
        pass


class RealTime(QScrollArea):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setStyleSheet(f"background-color: {colors['white_smoke']}; border: none;")
        self.configurations = Configurations()
        self.selected_pipeline_path = ""
        
        self.loaded_pipeline = None
        self.feature_mask = None
        self.target_channels = None
        
        self.baseline_features_buffer = []
        self.b_mean = None
        self.b_std = None
        self.is_calibrated = False
        self.baseline_target_sec = 0

        self.bci_interface = None
        self.acquisition_thread = None
        self.is_running = False
        self.dummy_viewer = DummyPromptViewer()
        
        self.mock_baseline_path = ""
        self.mock_test_path = ""
        self.is_mock_enabled = False
        self.stream_ch_names = []
        self.stream_sfreq = 250.0
        
        self.plot_x = deque(maxlen=100)
        self.plot_y = deque(maxlen=100)
        self.plot_y_raw = deque(maxlen=100)
        self.plot_current_time = 0
        
        self.prediction_signaler = ThreadSafePredictionSignaler(self.update_prediction_labels)

        content_widget = QWidget()
        self.setWidget(content_widget)

        self.layout = QVBoxLayout(content_widget)
        self.layout.setAlignment(Qt.AlignTop)

        app_title = TitleLabel("EEG Processor")
        self.layout.addWidget(app_title)

        back_btn = BackButton(controller)
        self.layout.addWidget(back_btn)

        page_title = QLabel("Real-Time Probability Plot")
        page_title.setStyleSheet(f"color: {colors['text']}; font-size: 20px; font-weight: bold; margin-top: 10px; margin-bottom: 20px;")
        self.layout.addWidget(page_title)

        file_row = QHBoxLayout()
        self.layout.addLayout(file_row)

        self.select_btn = QPushButton("Select Trained Pipeline File")
        self.select_btn.clicked.connect(self.select_pipeline_file)
        self.select_btn.setStyleSheet(f"QPushButton {{ background-color: {colors['surface']}; color: {colors['text']}; border: 1px solid {colors['border']}; border-radius: 4px; padding: 8px 15px; font-weight: bold; }} QPushButton:hover {{ background-color: {colors['border']}; }} QPushButton:disabled {{ background-color: #1a1a1a; color: #555555; border: 1px solid #222222; }}")
        file_row.addWidget(self.select_btn)

        self.pipeline_label = QLabel("No pipeline file selected")
        self.pipeline_label.setStyleSheet("padding: 8px; color: #aaaaaa; font-weight: normal;")
        file_row.addWidget(self.pipeline_label)
        file_row.addStretch()

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("padding: 8px; color: #2196F3; font-weight: bold;")
        self.layout.addWidget(self.status_label)
        
        base_title = QLabel("1. Baseline Calibration")
        base_title.setStyleSheet(f"color: {colors['text']}; font-weight: bold; margin-top: 10px;")
        self.layout.addWidget(base_title)

        base_layout = QHBoxLayout()
        self.live_baseline_spinbox = QSpinBox()
        self.live_baseline_spinbox.setRange(10, 300)
        self.live_baseline_spinbox.setValue(60)
        self.live_baseline_spinbox.setStyleSheet(f"background-color: {colors['surface']}; color: {colors['text']}; padding: 5px;")
        
        spin_label = QLabel("Gather live for (sec):")
        spin_label.setStyleSheet(f"color: {colors['text']};")
        base_layout.addWidget(spin_label)
        base_layout.addWidget(self.live_baseline_spinbox)
        
        or_label = QLabel("  OR  ")
        or_label.setStyleSheet(f"color: {colors['text']}; font-weight: bold;")
        base_layout.addWidget(or_label)
        
        self.mock_base_btn = QPushButton("Select Pre-recorded Baseline EDF")
        self.mock_base_btn.clicked.connect(self.select_mock_baseline)
        self.mock_base_btn.setStyleSheet(self.select_btn.styleSheet())
        base_layout.addWidget(self.mock_base_btn)

        self.clear_base_btn = QPushButton("Clear")
        self.clear_base_btn.clicked.connect(self.clear_mock_baseline)
        self.clear_base_btn.setEnabled(False)
        self.clear_base_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; border: none; border-radius: 4px; padding: 8px 15px; font-weight: bold; } QPushButton:hover { background-color: #d32f2f; } QPushButton:disabled { background-color: #333333; color: #888888; }")
        base_layout.addWidget(self.clear_base_btn)
        
        self.mock_base_label = QLabel("No file selected")
        self.mock_base_label.setStyleSheet("color: #aaaaaa; padding-left: 8px;")
        base_layout.addWidget(self.mock_base_label)
        base_layout.addStretch()
        self.layout.addLayout(base_layout)

        test_title = QLabel("2. Test Data Source")
        test_title.setStyleSheet(f"color: {colors['text']}; font-weight: bold; margin-top: 10px;")
        self.layout.addWidget(test_title)
        
        self.mock_checkbox = QCheckBox("Use MOCKED Data (Offline Analysis instead of Live BCI)")
        self.mock_checkbox.setStyleSheet(f"color: {colors['text']}; font-size: 14px;")
        self.mock_checkbox.stateChanged.connect(self.on_mock_toggled)
        self.layout.addWidget(self.mock_checkbox)
        
        mock_test_row = QHBoxLayout()
        self.mock_test_btn = QPushButton("Select Mock Test EDF (Task)")
        self.mock_test_btn.clicked.connect(self.select_mock_test)
        self.mock_test_btn.setEnabled(False)
        self.mock_test_btn.setStyleSheet(self.select_btn.styleSheet())
        mock_test_row.addWidget(self.mock_test_btn)
        
        self.mock_test_label = QLabel("No test file selected")
        self.mock_test_label.setStyleSheet("color: #aaaaaa; padding-left: 8px;")
        mock_test_row.addWidget(self.mock_test_label)
        mock_test_row.addStretch()
        self.layout.addLayout(mock_test_row)

        self.start_acquisition_button = QPushButton("Start Real Time Experiment")
        self.start_acquisition_button.clicked.connect(self.start_or_stop_acquisition)
        self.start_acquisition_button.setEnabled(False)
        self.start_acquisition_button.setStyleSheet("QPushButton { margin-top: 15px; padding: 12px; background-color: #2196F3; color: white; border: none; border-radius: 4px; font-weight: bold; font-size: 14px;} QPushButton:hover { background-color: #0b7dda; } QPushButton:disabled { background-color: #333333; color: #888888; }")
        self.layout.addWidget(self.start_acquisition_button)

        self.prediction_label = QLabel("Predicted Label: Waiting...")
        self.prediction_label.setStyleSheet("padding: 8px; color: #4CAF50; font-weight: bold; font-size: 18px;")
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.prediction_label)
        
        self.probabilities_label = QLabel("Probabilities: N/A")
        self.probabilities_label.setStyleSheet("padding: 8px; color: white; font-weight: normal; font-size: 14px;")
        self.probabilities_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.probabilities_label)

        self.plot_container = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_container)
        self.layout.addWidget(self.plot_container)
        self.init_probability_plot()

    def init_probability_plot(self):
        self.fig = Figure(figsize=(10, 4), facecolor='#121212')
        self.ax = self.fig.subplots()
        self.ax.set_facecolor('#1e1e1e')
        self.ax.set_title("Probability & Raw Decision Function", color='white', fontsize=12)
        
        self.ax.set_xlabel("Time (steps)", color='white')
        self.ax.set_ylabel("Probability (%)", color='#4CAF50')
        self.ax.tick_params(colors='#4CAF50')
        for spine in self.ax.spines.values(): spine.set_color('#2d2d2d')
        self.ax.set_ylim(-5, 105)
        self.ax.set_xlim(0, 100)
        self.ax.grid(True, color='#2d2d2d', linestyle='--')
        
        self.ax2 = self.ax.twinx()
        self.ax2.set_ylabel("Raw Logits", color='#2196F3')
        self.ax2.tick_params(colors='#2196F3')
        for spine in self.ax2.spines.values(): spine.set_color('#2d2d2d')
        
        self.line, = self.ax.plot([], [], color='#4CAF50', linewidth=2, label="Probability (0-100%)")
        self.line_raw, = self.ax2.plot([], [], color='#2196F3', linewidth=2, linestyle='--', label="Raw Output (Logits)")
        
        lines_1, labels_1 = self.ax.get_legend_handles_labels()
        lines_2, labels_2 = self.ax2.get_legend_handles_labels()
        self.ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", facecolor='#121212', edgecolor='#2d2d2d', labelcolor='white')
        
        self.fill = None
        self.plot_canvas = FigureCanvas(self.fig)
        self.plot_layout.addWidget(self.plot_canvas)

    def select_pipeline_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select Trained Pipeline File", "", "Model files (*.joblib *.pkl)")
        if filename:
            self.selected_pipeline_path = filename
            self.load_pipeline_file(filename)

    def load_pipeline_file(self, filepath):
        try:
            self.status_label.setStyleSheet("color: #2196F3; font-weight: bold; padding: 4px;")
            self.status_label.setText("Loading pipeline...")

            if filepath.endswith('.joblib'):
                pipeline = joblib.load(filepath)
                base_dir = os.path.dirname(filepath)
                base_name = os.path.basename(filepath).replace('_model.joblib', '')
                mask_path = os.path.join(base_dir, f"{base_name}_mask.joblib")
                channels_path = os.path.join(base_dir, f"{base_name}_channels.joblib")
                
                if os.path.exists(mask_path):
                    self.feature_mask = joblib.load(mask_path)
                else:
                    self.feature_mask = None
                    
                if os.path.exists(channels_path):
                    self.target_channels = joblib.load(channels_path)
                else:
                    self.target_channels = None
            else:
                with open(filepath, 'rb') as f:
                    pipeline = pickle.load(f)

            if not hasattr(pipeline, 'predict'): raise ValueError("Invalid estimator.")

            self.loaded_pipeline = pipeline
            channels_info = f" ({len(self.target_channels)} channels)" if self.target_channels else ""
            self.pipeline_label.setText(f"Loaded: {pipeline.__class__.__name__}{channels_info}")
            self.pipeline_label.setStyleSheet("padding: 8px; color: #4CAF50; font-weight: bold;")
            self.status_label.setText("Pipeline loaded successfully.")
            self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold; padding: 4px;")
            self.validate_start_button()

        except Exception as e:
            self.loaded_pipeline = None
            self.feature_mask = None
            self.target_channels = None
            self.pipeline_label.setText("No pipeline file selected")
            self.pipeline_label.setStyleSheet("padding: 8px; color: #aaaaaa;")
            self.validate_start_button()
            self.status_label.setText(f"Error loading pipeline: {e}")
            self.status_label.setStyleSheet("color: #f44336; font-weight: bold; padding: 4px;")

    def on_mock_toggled(self, state):
        self.is_mock_enabled = (state == Qt.Checked.value)
        self.mock_test_btn.setEnabled(self.is_mock_enabled)
        self.validate_start_button()
        
    def select_mock_baseline(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select Pre-recorded BASELINE EDF", "", "EDF files (*.edf)")
        if filename:
            self.mock_baseline_path = filename
            self.mock_base_label.setText(os.path.basename(filename))
            self.live_baseline_spinbox.setEnabled(False)
            self.clear_base_btn.setEnabled(True)
            self.validate_start_button()

    def clear_mock_baseline(self):
        self.mock_baseline_path = ""
        self.mock_base_label.setText("No file selected")
        self.live_baseline_spinbox.setEnabled(True)
        self.clear_base_btn.setEnabled(False)
        self.validate_start_button()

    def select_mock_test(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select Mock TEST EDF", "", "EDF files (*.edf)")
        if filename:
            self.mock_test_path = filename
            self.mock_test_label.setText(os.path.basename(filename))
            self.validate_start_button()
            
    def validate_start_button(self):
        can_start = self.loaded_pipeline is not None
        if self.is_mock_enabled:
            if not self.mock_test_path:
                can_start = False
        self.start_acquisition_button.setEnabled(can_start)

    def on_show(self): pass

    def on_hide(self):
        if self.is_running:
            self.start_or_stop_acquisition()

    def update_prediction_labels(self, label, probs_text, prob_value, raw_value):
        self.prediction_label.setText(f"Predicted Label: {label}")
        
        raw_info = f" | Raw Logits: {raw_value:.3f}" if raw_value is not None else ""
        self.probabilities_label.setText(f"Probabilities: {probs_text}{raw_info}")
        
        if prob_value >= 0.0:
            self.plot_x.append(self.plot_current_time)
            self.plot_y.append(prob_value * 100.0)
            
            if raw_value is not None:
                self.plot_y_raw.append(raw_value)
            
            self.plot_current_time += 1
            
            self.line.set_data(self.plot_x, self.plot_y)
            if self.plot_y_raw:
                self.line_raw.set_data(self.plot_x, self.plot_y_raw)
            
            if self.fill: self.fill.remove()
            self.fill = self.ax.fill_between(self.plot_x, 0, self.plot_y, color='#4CAF50', alpha=0.2)
            
            x_max = max(100, self.plot_current_time)
            self.ax.set_xlim(x_max - 100, x_max)
            
            if self.plot_y_raw:
                max_abs_val = max(abs(min(self.plot_y_raw)), abs(max(self.plot_y_raw)))
                padding = 0.5 if max_abs_val < 1.0 else max_abs_val * 0.2
                limit = max_abs_val + padding
                self.ax2.set_ylim(-limit, limit)
            
            self.plot_canvas.draw_idle()

    def enforce_volts(self, data):
        if np.std(data) > 1e-3: 
            return data * 1e-6
        return data

    def extract_features(self, epoch_data: np.ndarray, sfreq: float) -> np.ndarray:
        n_channels = epoch_data.shape[0]
        nperseg = min(epoch_data.shape[1], int(sfreq * 2))
        freqs, psd = welch(epoch_data, fs=sfreq, nperseg=nperseg, axis=-1)
        bands = {'delta': (0.5, 4.0), 'theta': (4.0, 8.0), 'alpha': (8.0, 13.0), 'beta': (13.0, 30.0), 'gamma': (30.0, 45.0)}
        
        features, band_powers = [], {}
        
        for name, (f_min, f_max) in bands.items():
            mask = (freqs >= f_min) & (freqs <= f_max)
            p_band = np.trapz(psd[:, mask], freqs[mask], axis=-1) if np.any(mask) else np.zeros(n_channels)
            band_powers[name] = p_band
            features.extend(p_band)
            features.extend(0.5 * np.log2(2 * np.pi * np.e * (p_band + 1e-12)))
            
        features.extend(band_powers['theta'] / (band_powers['beta'] + 1e-12))
        features.extend(band_powers['alpha'] / (band_powers['beta'] + 1e-12))
        
        return np.array(features)

    def start_or_stop_acquisition(self):
        if not self.is_running:
            self.is_running = True
            
            self.baseline_features_buffer = []
            self.b_mean = None
            self.b_std = None
            self.is_calibrated = False
            self.baseline_target_sec = self.live_baseline_spinbox.value()

            self.plot_x.clear()
            self.plot_y.clear()
            self.plot_y_raw.clear()
            self.plot_current_time = 0
            self.line.set_data([], [])
            self.line_raw.set_data([], [])
            if self.fill: 
                self.fill.remove()
                self.fill = None
            self.ax.set_xlim(0, 100)
            self.ax2.set_ylim(-1, 1)
            self.plot_canvas.draw_idle()
            
            self.dummy_viewer.closed = False
            
            if self.mock_baseline_path and self.feature_mask is not None:
                self.precalculate_mock_baseline()
            
            if self.is_mock_enabled:
                raw_test = mne.io.read_raw_edf(self.mock_test_path, preload=False, verbose=False)
                self.stream_sfreq = raw_test.info['sfreq']
                ch_names = raw_test.ch_names
                ch_names = [ch.replace('EEG ', '').replace('-REF', '').strip() for ch in ch_names]
                hardware_mapping = {
                    'FC5': 'F3', 'FC6': 'F4', 'CP5': 'P3', 'CP6': 'P4',
                    'FP1': 'Fp1', 'FP2': 'Fp2', 'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz', 'POZ': 'POz',
                    'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'
                }
                self.stream_ch_names = [hardware_mapping.get(ch, ch) for ch in ch_names]
                
                if self.is_calibrated or self.feature_mask is None:
                    self.queue = [["Mock Task Streaming", 999999.0]]
                else:
                    self.queue = [["baseline", float(self.baseline_target_sec)], ["Mock Task Streaming", 999999.0]]
            else:
                if self.bci_interface is None:
                    self.bci_interface = GtecNautilusProInterface()
                
                self.stream_sfreq = self.configurations.read("general.sampling_rate") or 250.0
                self.stream_ch_names = [
                    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6',
                    'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3',
                    'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', 'AF7', 'AF8'
                ]
                
                if self.is_calibrated or self.feature_mask is None:
                    self.queue = [["Real-Time Task Active", 999999.0]]
                else:
                    self.queue = [
                        ["baseline", float(self.baseline_target_sec)], 
                        ["Real-Time Task Active", 999999.0]
                    ]
            
            self.start_acquisition_button.setText("Stop Experiment")
            self.start_acquisition_button.setStyleSheet("QPushButton { margin-top: 15px; padding: 12px; background-color: #f44336; color: white; border: none; border-radius: 4px; font-weight: bold; font-size: 14px;} QPushButton:hover { background-color: #d32f2f; }")
            self.acquisition_thread = Thread(target=self.acquisition, daemon=True)
            self.acquisition_thread.start()
        else:
            self.is_running = False
            self.dummy_viewer.closed = True
            self.start_acquisition_button.setText("Start Real Time Experiment")
            self.start_acquisition_button.setStyleSheet("QPushButton { margin-top: 15px; padding: 12px; background-color: #2196F3; color: white; border: none; border-radius: 4px; font-weight: bold; font-size: 14px;} QPushButton:hover { background-color: #0b7dda; }")
            self.prediction_label.setText("Experiment Stopped")
            self.probabilities_label.setText("N/A")

    def acquisition(self):
        dummy_signaler = DummyTimelineSignaler()
        batches_per_second = 10
        
        try:
            if self.is_mock_enabled:
                mock_interface = MockBCIInterface(self.mock_test_path)
                mock_interface.run_acquisition(
                    self.dummy_viewer, self.queue, dummy_signaler, 0.0,
                    batches_per_second, real_time_processor=self.real_time_processor
                )
            else:
                self.bci_interface.run_acquisition(
                    self.dummy_viewer, self.queue, dummy_signaler, 0.0,
                    batches_per_second, real_time_processor=self.real_time_processor
                )
        except Exception as e:
            print("Acquisition error:", e)
            traceback.print_exc()
            
        self.is_running = False
        self.dummy_viewer.closed = True
        self.start_acquisition_button.setText("Start Real Time Experiment")
        self.start_acquisition_button.setStyleSheet("QPushButton { margin-top: 15px; padding: 12px; background-color: #2196F3; color: white; border: none; border-radius: 4px; font-weight: bold; font-size: 14px;} QPushButton:hover { background-color: #0b7dda; }")

    def precalculate_mock_baseline(self):
        self.status_label.setText("Extracting Baseline Features...")
        try:
            raw_base = mne.io.read_raw_edf(self.mock_baseline_path, preload=True, verbose=False)
            
            raw_base.rename_channels({ch: ch.replace('EEG ', '').replace('-REF', '').strip() for ch in raw_base.ch_names})
            hardware_mapping = {
                'FC5': 'F3', 'FC6': 'F4', 'CP5': 'P3', 'CP6': 'P4',
                'FP1': 'Fp1', 'FP2': 'Fp2', 'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz', 'POZ': 'POz',
                'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'
            }
            raw_base.rename_channels({ch: hardware_mapping.get(ch, ch) for ch in raw_base.ch_names})
            
            sfreq = raw_base.info['sfreq']
            
            if self.target_channels:
                avail_ch = [ch for ch in self.target_channels if ch in raw_base.ch_names]
                raw_base.pick(avail_ch)
                raw_base.reorder_channels(avail_ch)  
                
            data_raw = self.enforce_volts(raw_base.get_data())
            
            trial_length = 2.0
            required_samples = int(sfreq * trial_length)
            step = int(sfreq * 1.0)
            
            is_feature_based = (self.feature_mask is not None)
            
            baseline_feats = []
            for start in range(0, data_raw.shape[1] - required_samples, step):
                window_raw = data_raw[:, start:start+required_samples]
                
                if is_feature_based:
                    window_step = window_raw - np.mean(window_raw, axis=0, keepdims=True)
                else:
                    window_step = window_raw
                    
                window_filtered = mne.filter.filter_data(window_step, sfreq=sfreq, l_freq=4.0, h_freq=45.0, method='iir', verbose=False)
                baseline_feats.append(self.extract_features(window_filtered, sfreq))
                
            if baseline_feats:
                self.b_mean = np.mean(baseline_feats, axis=0)
                self.b_std = np.std(baseline_feats, axis=0) + 1e-12
                self.is_calibrated = True
                self.status_label.setText(f"Baseline Calibrated ({len(baseline_feats)} epochs). Ready.")
            else:
                self.status_label.setText("Error: Baseline file too short.")
        except Exception as e:
            self.status_label.setText(f"Baseline error: {e}")

    def real_time_processor(self, current_signal, current_true_label=None):
        if not self.loaded_pipeline: return "No pipeline loaded"
            
        sampling_rate = self.stream_sfreq
        trial_length = 2.0 
        required_samples = int(sampling_rate * trial_length)
        
        current_signal = np.array(current_signal)
        if current_signal.shape[1] < required_samples: return "Gathering data..."
            
        raw_data = current_signal[:, -required_samples:]
        
        if self.target_channels:
            target_indices = []
            for ch in self.target_channels:
                if ch in self.stream_ch_names:
                    target_indices.append(self.stream_ch_names.index(ch))
            process_data_raw = raw_data[target_indices, :]
        else:
            process_data_raw = raw_data

        process_data_raw = self.enforce_volts(process_data_raw)
        
        is_feature_based = (self.feature_mask is not None)

        if is_feature_based:
            process_data_step = process_data_raw - np.mean(process_data_raw, axis=0, keepdims=True)
        else:
            process_data_step = process_data_raw
            
        process_data_filt = mne.filter.filter_data(process_data_step, sfreq=sampling_rate, l_freq=4.0, h_freq=45.0, method='iir', verbose=False)
        
        prob_val = -1.0 
        raw_val = 0.0

        label_norm = str(current_true_label).lower() if current_true_label else ""
        
        if "rest" in label_norm or "baseline" in label_norm or "natural" in label_norm:
            if is_feature_based:
                raw_features = self.extract_features(process_data_filt, sampling_rate)
                self.baseline_features_buffer.append(raw_features)
            else:
                self.baseline_features_buffer.append(1)
                
            collected_sec = len(self.baseline_features_buffer) / 10.0
            status_text = f"Please relax... ({collected_sec:.1f}s / {self.baseline_target_sec}s)" if self.baseline_target_sec > 0 else f"Please relax... ({collected_sec:.1f}s)"
            
            if hasattr(self, 'prediction_signaler'):
                self.prediction_signaler("Calibrating Baseline...", status_text, -1.0, 0.0)
            return "calibrating"

        try:
            if is_feature_based:
                raw_features = self.extract_features(process_data_filt, sampling_rate)
                
                if not self.is_calibrated and len(self.baseline_features_buffer) > 0:
                    buffer_arr = np.array(self.baseline_features_buffer)
                    self.b_mean = np.mean(buffer_arr, axis=0)
                    self.b_std = np.std(buffer_arr, axis=0) + 1e-12
                    self.is_calibrated = True
                
                if not self.is_calibrated:
                    self.b_mean = np.zeros_like(raw_features)
                    self.b_std = np.ones_like(raw_features)
                    self.is_calibrated = True

                norm_features = (raw_features - self.b_mean) / self.b_std
                final_features = norm_features[self.feature_mask]
                X_test = np.expand_dims(final_features, axis=0)
            else:
                self.is_calibrated = True
                X_test = np.expand_dims(process_data_filt, axis=0)
                
            prediction = self.loaded_pipeline.predict(X_test)[0]
            pred_str = "Concentration" if int(prediction) == 1 else "Rest"
            
            prob_str = "N/A"
            if hasattr(self.loaded_pipeline, "predict_proba"):
                probs = self.loaded_pipeline.predict_proba(X_test)[0]
                prob_val = probs[1]
                prob_str = f"Concentration: {probs[1]*100:.1f}% | Rest: {probs[0]*100:.1f}%"
                
            if hasattr(self.loaded_pipeline, "decision_function"):
                raw_val = float(self.loaded_pipeline.decision_function(X_test)[0])
                
        except Exception:
            return "Error"
            
        if hasattr(self, 'prediction_signaler'):
            self.prediction_signaler(pred_str, prob_str, prob_val, raw_val)
            
        return pred_str