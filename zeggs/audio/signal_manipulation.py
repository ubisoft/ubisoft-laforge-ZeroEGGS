import scipy.signal as sps


def preemphasis(x, preemph=0.97):
    ''' Perform high pass filtering on input signal

    :param x:           signal to filter
    :param preemph:     pre-emphasis factor

    :return: high pass filtered signal
    '''
    return sps.lfilter([1, -preemph], [1], x)
