import scipy


class filter:
    """ Docstring...

        Abstract filter class

        Attributes:
            

        Methods:  
            apply_filter(nparray):
                this method is called to apply the filter to the given signal.
                it returns the filtered signal.
            
            _filter(nparraz):
                this method is called by apply_filter to apply the filter to the given signal.
                it returns the filtered signal.
                this mehtod has to be implemented by the child class.

    """
        
    def apply_filter(self, signal):
        return self._filter(signal)
    
    def _filter(signal):
        pass
    
class bandpass_filter(filter):
    wn_EEG = [0.2, 35]
    wn_EOG = [0.2, 10]

    def __init__(self, wn, fs=250) -> None:
        self.wn = wn
        self.fs = fs
    
    def _filter(self, signal):
        N = 2**3 # Orden del filtro
        sos = scipy.signal.butter(N, self.wn, btype = 'bandpass', output = 'sos', fs = self.fs) # Diseño el filtro
        return scipy.signal.sosfiltfilt(sos, signal) # Filtro la señal
