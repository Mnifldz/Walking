import numpy as np

from scipy.signal import butter, lfilter, medfilt


def butter_lowpass_filter(data, corner=0.02, fs=50, order=3):
    """
    Uses signal.butter to perform the filter on data
    Arguments:
        <data> (numpy array) --- one dimensional timeseries data
        which the filter is to be applied to
        <corner> (float) --- critical (corner) frequency for LPF
        <fs> (float) --- sampling rate
        <order> (int) --- represents the order of the filter
    Returns:
        <y> (numpy array) --- data which the filter has been applied to
    """

    # scipy.signal.butter accepts an argument "Wn is normalized from 0 to 1,
    # where 1 is the Nyquist frequency, pi radians/sample"
    # the nyquist frequency is fs/2, so we must express Wn as corner / (0.5 * fs)

    Wn = corner / (0.5 * fs)
    b, a = butter(order, Wn, 'low')
    
    return lfilter(b, a, data)


def get_gravity(data):
    '''
    Takes in the raw accelerometer data as a n x 3 numpy array.
    Columns x,y,z
    Implements 2/3 of the technique shown in the following paper:
    https://hal-upec-upem.archives-ouvertes.fr/hal-00826243/document
    
    last 1/3 may be needed
    
    https://github.com/KalebKE/AccelerationExplorer/wiki/Low-Pass-Filter-Linear-Acceleration
    
    '''
    data = np.apply_along_axis(butter_lowpass_filter, 0, data)
    data = np.apply_along_axis(medfilt, 0, data)
    return data