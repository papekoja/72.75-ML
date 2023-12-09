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
        self.ch_names = None

class sleepRecording:
    """ Docstring...

        Represents a sleep recording session and provides methods to initialize
        the session from PSG and hypnogram files, extract epochs, and print
        information about the recording.

        Attributes:
            epochs (list): A list to store individual sleep epochs.
            epoch_duration (int): The duration of each sleep epoch in seconds.
            freq (float): The frequency of the recording in Hz.
            recording_duration_sec (float): The total duration of the recording in seconds.
            order_index (int): An index to track the order of sleep recordings.

        Methods:
            init_from_file(path_psg, path_hyp):
                Initializes the sleep recording session from PSG and hypnogram files.

            print_duration():
                Prints the total recording duration in seconds, minutes, and hours.
    """


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

            # TODO - Work further here, I want to add other variables. I will like ch_names and i would like to split the datas into its different channels. I have no idea how now but eventually, some kind of feature extraction will have to take place over these channels. 

            # Other variables i guess are nice
            # self.ch_names = psg_data.info.get('ch_names')       # maybe channel names is just nice as a reference is needed, doesn't work in current stat, need to fix
            self.EEG_Fpz_Cz = psg_data[0]           # I try to get the different channels here but idk if it works

        print('check 1')

    def print_duration(self):
        print(f'{self.recording_duration_sec} Seconds')
        print(f'{self.recording_duration_sec/60} Minutes')
        print(f'{round(self.recording_duration_sec/60/60, 2)} Hours')


if __name__ == "__main__":
    s = sleepRecording()
    s.init_from_file("data/SC4001E0-PSG.edf","data/SC4001EC-Hypnogram.edf")
    print("a")
    for x in s.epochs:
        print(x.label)
