import mne
import time
import numpy as np
from datetime import datetime

from config.config import Configurations

class MockBCIInterface:
    def __init__(self, edf_path):
        self.edf_path = edf_path
        self.configurations = Configurations()
        
    def run_acquisition(self, prompt_viewer, current_queue, update_experiment_timeline_plot, progressbar_value, batches_per_second, real_time_processor=None):
        raw = mne.io.read_raw_edf(self.edf_path, preload=True, verbose=False)
        
        # Standaryzacja nazw kanałów
        raw.rename_channels({ch: ch.replace('EEG ', '').replace('-REF', '').strip() for ch in raw.ch_names})
        
        # KOMPLETNE MAPOWANIE SPRZĘTOWE
        hardware_mapping = {
            'FC5': 'F3', 'FC6': 'F4', 'CP5': 'P3', 'CP6': 'P4',
            'FP1': 'Fp1', 'FP2': 'Fp2', 'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz', 'POZ': 'POz',
            'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'
        }
        raw.rename_channels({ch: hardware_mapping.get(ch, ch) for ch in raw.ch_names})
        
        data = raw.get_data() # shape: (channels, times)
        sfreq = int(raw.info['sfreq'])
        channel_names = raw.ch_names
        
        batch_size = sfreq // batches_per_second
        total_samples = data.shape[1]
        
        # Omijamy sztywne 32 kanały G.tec - dynamicznie przydzielamy tyle list, ile jest elektrod w EDF
        signal = [[] for _ in range(len(channel_names))]
        start_date = datetime.now()
        
        for start_idx in range(0, total_samples, batch_size):
            if current_queue is None or len(current_queue) == 0:
                break
                
            end_idx = min(start_idx + batch_size, total_samples)
            if start_idx == end_idx:
                break
                
            chunk = data[:, start_idx:end_idx] 
            
            # Sygnał leci w naturalnej, niezmienionej kolejności
            for channel in range(len(channel_names)):
                signal[channel] = np.concatenate((signal[channel], list(chunk[channel, :])))
                    
            item = current_queue[0]
            prompt_to_show = item[0]
            
            if real_time_processor is not None:
                prompt_text = real_time_processor(signal, current_true_label=prompt_to_show)
                if prompt_text is not None:
                    prompt_to_show = prompt_text
                    
            print('MOCK BCI', datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], len(signal[0]), prompt_to_show)
            
            if not prompt_viewer.closed:
                prompt_viewer.change_prompt(prompt_to_show)
                
                item[1] -= 1.0 / batches_per_second
                item[1] = round(item[1], 1)
                
                if item[1] < 0.1 and current_queue is not None:
                    current_queue.pop(0)
                    
                update_experiment_timeline_plot(1.0 / batches_per_second)
            else:
                break
                
            time.sleep(1.0 / batches_per_second)
            
        return signal, start_date