import matplotlib.pyplot as plt
import numpy as np
import datetime
import mne
import scipy
from mne.datasets.sleep_physionet.age import fetch_data


class epoch:
    def __init__(self) -> None:
        self.data = None
        self.data_time = None
        self.lable = None
        self.start = None
        self.end = None
        self.duration = None
        self.freq = None

class sleepRecording:
    def __init__(self, epoch_duration = 30) -> None:
        self.epochs = []
        self.epoch_duration = epoch_duration
        self.freq = None
        self.recording_duration_sec = None
        self.order_inex = 0
    
    def init_from_file(self, path_psg, path_hyp):
        #load files
        psg_data_raw = mne.io.read_raw_edf(path_psg)
        hypnogramm_annotations = mne.read_annotations(path_hyp)

        psg_data = psg_data_raw.get_data()
        psg_data_raw.set_annotations(hypnogramm_annotations)

        #get frequency and duration of recording
        self.freq = psg_data_raw.info.get('sfreq')
        self.recording_duration_sec = psg_data.shape[1]/self.freq

        for epoch_index in range(0, int(self.recording_duration_sec / self.epoch_duration)):
            epoch_ = epoch()
            epoch_.start = epoch_index * self.epoch_duration
            epoch_.end = (epoch_index + 1) * self.epoch_duration
            epoch_.duration = self.epoch_duration
            epoch_.freq = self.freq

            epoch_start = int(epoch_index * self.epoch_duration * self.freq)
            epoch_end = int((epoch_index + 1) * self.epoch_duration * self.freq) 

            start_datetime = psg_data_raw.info['meas_date']
            annotations_of_frame = psg_data_raw.annotations.copy().crop(
            start_datetime.timestamp() + epoch_index * self.epoch_duration,
            start_datetime.timestamp() + epoch_index * self.epoch_duration + self.epoch_duration)

            #find the most imortant label
            annotation_max = annotations_of_frame[0]
            annotation_max_duration = annotations_of_frame[0]['duration']
            for annotation_index in range(len(annotations_of_frame)):
                annotation = annotations_of_frame[annotation_index]
                if annotation['duration'] > annotation_max_duration:
                    annotation_max = annotation
                    annotation_max_duration = annotation['duration']

            epoch_.label = annotation_max['description']
            self.epochs.append(epoch_)

    def print_duration(self):
        print(f'{self.recording_duration_sec} segundos')
        print(f'{self.recording_duration_sec/60} minutos')
        print(f'{self.recording_duration_sec/60/60} horas')
    
s = sleepRecording()
s.init_from_file("data/SC4001E0-PSG.edf","data/SC4001EC-Hypnogram.edf")
print("a")
for x in s.epochs:
    print(x.label)
