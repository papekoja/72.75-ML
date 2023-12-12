import scipy
import epoch
import numpy as np

def create_feature_avg_temp(e: epoch.epoch):
    channle = e.get_channle_by_name("Temp rectal")
    e.features["avg_temp"] = np.mean(channle)

def create_feature_welsh(e: epoch.epoch):
    nper = int(e.freq*10)
    channle_0 = e.get_channle_by_name("EEG Pz-Oz")
    channle_1 = e.get_channle_by_name("EEG Fpz-Cz")
    _, e.features["welsh_Pz-Oz"] = scipy.signal.welch(channle_0*1e6, e.freq, noverlap=nper/2  , nperseg=nper)
    e.features["welsh_f"], e.features["welsh_Fpz-Cz"] = scipy.signal.welch(channle_1*1e6, e.freq, noverlap=nper/2  , nperseg=nper)

def create_features_recording_session(sleepRecording: epoch.sleepRecording):
    for e in sleepRecording.epochs:
        create_feature_avg_temp(e)
        create_feature_welsh(e)