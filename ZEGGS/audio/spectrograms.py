import numpy as np
import scipy.signal as sps

from .logs import get_logger_from_arg
from .signal_manipulation import preemphasis


def extract_mel_spectrogram_for_tts(wav_signal, fs, n_fft, step_size, n_mels, mel_fmin, mel_fmax, min_amplitude,
                                    pre_emphasis=True, pre_emph_coeff=0.97, dynamic_range=None, real_amplitude=True,
                                    centered=True, normalize_mel_bins=True, normalize_range=True, logger=None):
    """ Extract mel-spectrogram from an audio signal for TTS training

    :param wav_signal:          Numpy array of audio samples -- shape = (T, )
    :param fs:                  sampling frequency of the audio signal
    :param n_fft:               filter length (in samples) of the FFT
    :param step_size:           length (in samples) between successive analysis windows
    :param n_mels:              number of mel components in the mel-spectrogram
    :param mel_fmin:            minimum frequency used when converting to mel
    :param mel_fmax:            maximum frequency used when converting to mel
    :param min_amplitude:       mel-spectrogram minimal permitted amplitude value (limits the dynamic range)
    :param pre_emphasis:        perform pre-emphasis on input audio
    :param pre_emph_coeff:      pre-emphasis coefficient
    :param dynamic_range:       mel-spectrogram maximal dynamic range in dB (ignored if min_amplitude is specified)
    :param real_amplitude:      if True, the value of the spectrogram bins will be divided by n_fft to get bin magnitude that
                                reflect the temporal signal amplitude
    :param centered:            if True, the spectrogram extraction window will be centered on the time step.
                                The time sequence has to be padded.
    :param normalize_mel_bins:  normalize energy per bins in the mel-spectrogram
    :param normalize_range:     If True, map the db_dynamic_range to [0,1]
    :param logger:              arg to create logger object

    :return: the mel-spectrogram corresponding to the input audio
    """
    # perform pre-emphasis on input audio
    if pre_emphasis:
        wav_signal = preemphasis(wav_signal, preemph=pre_emph_coeff)

    # get linear amplitude spectrogram
    s, _ = extract_spectrogram(x=wav_signal, n_fft=n_fft, step_size=step_size,
                               real_amplitude=real_amplitude, centered=centered)

    # convert to mel frequency scale
    s = linear_to_mel(linear_spectrogram=s, fs=fs, n_mels=n_mels, mel_fmin=mel_fmin, mel_fmax=mel_fmax,
                      normalize_mel_bins=normalize_mel_bins, logger=logger)

    # extract min amplitude to clip the mel-spectrogram and set the dynamic range
    if min_amplitude or dynamic_range:
        min_amplitude = get_spectrogram_min_amplitude(real_amplitude=real_amplitude, min_amplitude=min_amplitude,
                                                      dynamic_range=dynamic_range, n_fft=n_fft, logger=logger)

    # convert to dB and normalize range to [0, 1]
    s = amplitude_to_db(spectrogram=s, min_amplitude=min_amplitude, normalize_range=normalize_range, logger=logger)

    return s, wav_signal


def get_spectrogram_min_amplitude(real_amplitude, min_amplitude=None, dynamic_range=None, n_fft=None, logger=None):
    """ Compute the minimum amplitude value a spectrogram bin can reach

    :param real_amplitude:  If True, assume that the values of the spectrogram bins were divided by n_fft to get
                            bin magnitude that reflect the temporal signal amplitude
    :param min_amplitude:   The spectrogram minimal permitted amplitude value (limits the dynamic range)
                            This value is affected when real_amplitude is set to True
    :param dynamic_range:   The spectrogram maximal dynamic range in dB (ignored if min_amplitude is specified)
                            This value is affected when real_amplitude is set to True
    :param n_fft:           Number of samples of the FFT window used to extract spectrogram
                            Only used when real_amplitude is set to True
    :param logger:          arg to create logger object

    :return: the minimum amplitude of spectrogram bins
    """
    # create logger object
    logger = get_logger_from_arg(logger)

    if min_amplitude and dynamic_range:
        logger.warning(f'Both "min_amplitude" and "dynamic_range" are specified, '
                       f'only "min_amplitude" ({min_amplitude}) will be considered')
    else:
        assert (min_amplitude or dynamic_range), logger.error(f'Neither "min_amplitude" nor "dynamic_range" are set')

    if real_amplitude:
        assert (n_fft is not None), logger.error(f'"real_amplitude" is set to True but "n_fft" has no value')
    else:
        n_fft = 1  # equivalent to using a FFT window of 1

    if min_amplitude:
        # compute real min amplitude per bin
        min_amplitude = min_amplitude / n_fft

    elif dynamic_range:
        # compute real dynamic range per bin
        dynamic_range = dynamic_range + 20 * np.log10(n_fft)
        # compute real min amplitude per bin
        min_amplitude = 10 ** (-dynamic_range / 20)

    return min_amplitude


def amplitude_to_db(spectrogram, min_amplitude=None, normalize_range=False, logger=None):
    """ Transform amplitude to dB with optional clipping and dynamic range normalization

    :param spectrogram:         Numpy array containing all amplitudes of a spectrogram
    :param min_amplitude:       Clip the spectrogram to the minimal permitted amplitude value
    :param normalize_range:     If True, map the db_dynamic_range to [0,1]
    :param logger:              arg to create logger object

    :return: spectrogram in dB
    """
    # create logger object
    logger = get_logger_from_arg(logger)

    # make sure amplitude bins are positive
    spectrogram = np.abs(spectrogram)

    if min_amplitude:
        # apply clipping
        spectrogram = np.clip(spectrogram, a_min=min_amplitude, a_max=None)

    # transform to dB
    spectrogram = 20 * np.log10(spectrogram)

    # normalize range if necessary
    if normalize_range:
        # min_amplitude must be given to normalize de dB dynamic range
        assert (min_amplitude), logger.error(f'Asked for dynamic range normalization, but "min_amplitude" has no value')

        # compute dB dynamic range and map it to [0, 1]
        dynamic_range = -20 * np.log10(min_amplitude)
        spectrogram = (spectrogram + dynamic_range) / dynamic_range

    return spectrogram


def denormalize_range(spectrogram, min_amplitude_used):
    """ Take a dB spectrogram that has been mapped between [0, 1] and shape it back to its original dB dynamic range

    :param spectrogram:         Numpy array containing all amplitudes of a spectrogram in dB (values between 0 and 1)
    :param min_amplitude_used:  Minimal amplitude value that was used to normalize the dB spectrogram dynamic range

    :return: spectrogram in dB with its range de-normalized
    """
    # compute dB dynamic range
    dynamic_range = -20 * np.log10(min_amplitude_used)

    # denormalize dB dynamic range
    spectrogram = spectrogram * dynamic_range - dynamic_range

    return spectrogram


def db_to_amplitude(spectrogram):
    """ Transform dB spectrogram to amplitude spectrogram

     :param spectrogram:    Numpy array containing all amplitude of a spectrogram

     :return: spectrogram in amplitude value
     """
    return 10 ** (spectrogram / 20)


def linear_to_mel(linear_spectrogram, fs=None, n_mels=80, mel_fmin=0, mel_fmax=None,
                  normalize_mel_bins=True, logger=None):
    """ Convert a linear spectrogram to a mel-spectrogram

    :param linear_spectrogram:    Numpy array containing all amplitudes of a spectrogram -- shape = (n_fft // 2 + 1, T)
    :param fs:                    Sampling frequency expected by the algorithm
    :param n_mels:                Number of bins in the mel-spectrogram
    :param mel_fmin:              Lowest frequency in the mel-spectrum (Hz)
    :param mel_fmax:              Highest frequency in the mel-spectrum (Hz)
    :param normalize_mel_bins:    normalize energy per bins in the mel-spectrogram
    :param logger:                arg to create logger object

    :return: Numpy array containing the spectrogram in mel frequency space -- shape = (n_mels, T)
    """
    # find the number of samples used in the FFT window
    n_fft = (linear_spectrogram.shape[0] - 1) * 2

    # get filter parameters
    mel_basis = _get_mel_filterbank_matrix(n_fft=n_fft, fs=fs, n_mels=n_mels, mel_fmin=mel_fmin, mel_fmax=mel_fmax,
                                           normalize_mel_bins=normalize_mel_bins, logger=logger)

    # apply filter bank matrix
    return np.dot(mel_basis, linear_spectrogram)


def mel_to_linear(mel_spectrogram, fs, n_fft, mel_fmin=0, mel_fmax=None, normalize_mel_bins=False, logger=None):
    """ Convert a mel-spectrogram to a linear spectrogram

    :param mel_spectrogram:         Numpy array of the input mel spectrogram -- shape = (n_mels, T)
    :param fs:                      sampling frequency
    :param n_fft:                   number of samples used in the original FFT
    :param mel_fmin:                minimum frequency used when converting to mel
    :param mel_fmax:                maximum frequency used when converting to mel
    :param normalize_mel_bins:      normalize energy per bins in the mel-spectrogram
    :param logger:                  arg to create logger object

    :return: Numpy array containing the spectrogram in linear frequency space -- shape = (n_fft // 2 + 1, T)
    """
    # find the number of mel components
    n_mels = mel_spectrogram.shape[0]

    # get filter parameters
    mel_basis = _get_mel_filterbank_matrix(n_fft=n_fft, fs=fs, n_mels=n_mels, mel_fmin=mel_fmin, mel_fmax=mel_fmax,
                                           normalize_mel_bins=normalize_mel_bins, logger=logger)

    # normalise the row of the mel_basis
    weight_value = mel_basis.sum(axis=1)
    mel_basis = np.divide(mel_basis, weight_value.reshape(n_mels, 1))

    # apply the inverse of the mel_filter bank to the algorithm
    linear_spectrogram = np.dot(np.transpose(mel_spectrogram), mel_basis)

    return np.transpose(linear_spectrogram)


def extract_spectrogram(x, n_fft, step_size, real_amplitude=True, centered=True):
    """ Extract the FFT spectrogram from a series of samples

    :param x:                   Numpy array of input samples -- shape = (T, )
    :param n_fft:               number of point in the FFT window
    :param step_size:           number of samples skipped at each extraction
    :param real_amplitude:      if True the value of the bins will be divided by n_fft to get bin magnitude that
                                reflect the temporal signal amplitude
    :param centered:            if True, the extraction window will be centered on the time step.
                                The time sequence has to be padded.

    :return: Numpy arrays of amplitude and phase of the spectrogram -- shapes = (n_fft // 2 + 1, L)
    """
    # create the sampling window
    window = sps.hann(n_fft)

    # check input signal has a length superior or equal to n_fft
    if len(x) < n_fft:
        x = np.pad(x, (0, len(window) - len(x)), 'constant', constant_values=(0, 0))

    # pad before and after to center the window on the extracted values
    if centered:
        padding_left, padding_right = _get_padding_for_centered_spectrogram(n_fft=n_fft)
        x = np.pad(x, (padding_left, padding_right), mode='reflect')

    # count the number of frames 
    if len(x) % step_size == 0:
        time_axis = int(np.floor((len(x) - n_fft) / step_size))
    else:
        time_axis = 1 + int(np.floor((len(x) - n_fft) / step_size))

    # create container for spectrogram
    amp = np.zeros((n_fft // 2 + 1, time_axis))
    phase = np.zeros((n_fft // 2 + 1, time_axis))

    for i in range(time_axis):
        # get slice of data
        win_data = x[i * step_size: i * step_size + n_fft]

        # apply windowing
        win_data = np.multiply(win_data, window)

        # get FFT
        freq = np.fft.rfft(win_data)

        # save magnitude and phase individually
        amp[:, i] = np.absolute(freq)
        phase[:, i] = np.angle(freq)

    # scale amplitude bins if necessary
    if real_amplitude:
        amp = amp / n_fft

    return amp, phase


def get_nb_spectrogram_samples(wav_length, n_fft, step_size, centered=True):
    """ Return the number of spectrogram time frames given a WAV segment

    :param wav_length:      number of samples in the WAV segment
    :param n_fft:           filter length (in samples) of the FFT
    :param step_size:       length (in samples) between successive analysis windows
    :param centered:        if True, assume that the FFT extraction window is centered on the time step

    :return: the number of spectrogram time frames
    """
    # create random signal
    random_signal = np.random.rand(wav_length)

    # extract amp and phase spectrograms -- shapes = (n_fft // 2 + 1, T)
    amp, phase = extract_spectrogram(x=random_signal, n_fft=n_fft, step_size=step_size, centered=centered)

    # return T
    return amp.shape[1]


def get_nb_wav_samples(spectrogram_length, n_fft, step_size, centered=True):
    ''' Return the number of WAV samples given a spectrogram segment

    :param spectrogram_length:      number of time frames in the spectrogram segment
    :param n_fft:                   filter length (in samples) of the FFT
    :param step_size:               length (in samples) between successive analysis windows
    :param centered:                if True, assume that the FFT extraction window is centered on the time step

    :return: the number of WAV samples
    '''
    # audio segment was padded on the left and right to center the window on the extracted values
    if centered:
        padding_left, padding_right = _get_padding_for_centered_spectrogram(n_fft=n_fft)
    else:
        padding_left, padding_right = 0, 0

    return (spectrogram_length - 1) * step_size + n_fft - padding_left - padding_right


def reconstruct_signal_griffin_lim(magnitude_spectrogram, step_size, iterations=30, logger=None):
    """ Reconstruct an audio signal from a magnitude spectrogram

        Given a magnitude spectrogram as input, reconstruct the audio signal and return it using
        the Griffin-Lim algorithm
        From the paper: "Signal estimation from modified short-time fourier transform" by Griffin and Lim, in IEEE
                        transactions on Acoustics, Speech, and Signal Processing. Vol ASSP-32, No. 2, April 1984.

    :param magnitude_spectrogram:   Numpy array magnitude spectrogram -- shape = (n_fft // 2 + 1, T)
                                    The rows correspond to frequency bins and the columns correspond to time slices
    :param step_size:               length (in samples) between successive analysis windows
    :param iterations:              Number of iterations for the Griffin-Lim algorithm
                                    Typically a few hundred is sufficient
    :param logger:                  arg to create logger object

    :return: the reconstructed time domain signal as a 1-dim Numpy array and the spectrogram that was used
             to produce the signal
    """
    # create logger object
    logger = get_logger_from_arg(logger)

    # shape = (T, n_fft // 2 + 1)
    magnitude_spectrogram = np.transpose(magnitude_spectrogram)

    # find the number of samples used in the FFT window and extract the time steps
    n_fft = (magnitude_spectrogram.shape[1] - 1) * 2
    time_slices = magnitude_spectrogram.shape[0]

    # compute the number of samples needed
    len_samples = int(time_slices * step_size + n_fft)

    # initialize the reconstructed signal to noise
    x_reconstruct = np.random.randn(len_samples)
    window = np.hanning(n_fft)
    n = iterations  # number of iterations of Griffin-Lim algorithm

    while n > 0:
        # decrement and compute FFT
        n -= 1
        reconstruction_spectrogram = np.array([np.fft.rfft(window * x_reconstruct[i: i + n_fft])
                                               for i in range(0, len(x_reconstruct) - n_fft, step_size)])

        # Discard magnitude part of the reconstruction and use the supplied magnitude spectrogram instead
        proposal_spectrogram = magnitude_spectrogram * np.exp(1.0j * np.angle(reconstruction_spectrogram))

        # store previous reconstructed signal and create a new one by iFFT
        prev_x = x_reconstruct
        x_reconstruct = np.zeros(len_samples)

        for i, j in enumerate(range(0, len(x_reconstruct) - n_fft, step_size)):
            x_reconstruct[j: j + n_fft] += window * np.real(np.fft.irfft(proposal_spectrogram[i]))

        # normalise signal due to overlap add
        x_reconstruct = x_reconstruct / (n_fft / step_size / 2)

        # compute diff between two signals and report progress
        diff = np.sqrt(sum((x_reconstruct - prev_x) ** 2) / x_reconstruct.size)
        logger.debug(f'Reconstruction iteration: {iterations - n}/{iterations} -- RMSE: {diff * 1e6:.3f}e-6')

    return x_reconstruct, proposal_spectrogram


def _get_padding_for_centered_spectrogram(n_fft):
    """ Return padding that must be added to the left and right sides of a series of samples to extract a centered FFT

    :param n_fft:       filter length (in samples) of the FFT

    :return: padding values for left and right sides
    """
    # add same padding on left and right sides
    padding_left, padding_right = int(n_fft // 2), int(n_fft // 2)

    return padding_left, padding_right


def _get_mel_filterbank_matrix(n_fft=None, fs=None, n_mels=80, mel_fmin=0, mel_fmax=None,
                               normalize_mel_bins=False, logger=None):
    """ Create a Filterbank matrix to combine FFT bins into Mel-frequency bins

    :param n_fft:               number of FFT components
    :param fs:                  sampling rate of the incoming signal
    :param n_mels:              number of Mel bands to generate
    :param mel_fmin:            lowest frequency (in Hz)
    :param mel_fmax:            highest frequency (in Hz). If None, mel_fmax = sr / 2.0
    :param normalize_mel_bins:  normalize energy per bins
    :param logger:              arg to create logger object

    :return: np.ndarray [shape=(n_mels, 1 + n_fft // 2)] -- Mel transform matrix
    """
    # create logger object
    logger = get_logger_from_arg(logger)

    # set mel_fmax
    if mel_fmax is None:
        mel_fmax = float(fs) / 2

    # Initialize the weights
    weights = np.zeros((int(n_mels), int(1 + n_fft // 2)))

    # Get the center frequencies of each FFT bin
    fft_freqs = np.linspace(0, float(fs) / 2, int(1 + n_fft // 2), endpoint=True)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = _hz_to_mel(mel_fmin)
    max_mel = _hz_to_mel(mel_fmax)

    mels = np.linspace(min_mel, max_mel, n_mels + 2)
    mel_f = _mel_to_hz(mels)

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fft_freqs)

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]

        # then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    if normalize_mel_bins:  # Normalize energy per bins
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]

    # Only check weights if f_mel[0] is positive
    if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):  # This means we have an empty channel somewhere
        # create logger object (only if needed)
        logger = get_logger_from_arg(logger)
        logger.warning('Empty filters detected in mel frequency basis. Some channels will produce empty responses. '
                       'Try increasing your sampling rate (and fmax) or reducing n_mels.')

    return weights


def _hz_to_mel(frequencies):
    """ Convert Hz to Mels

    :param frequencies:     number or np.ndarray [shape=(n,)] -- scalar or array of frequencies

    :return: number or np.ndarray [shape=(n,)] -- input frequencies in Mels
    """
    # create frequencies array
    frequencies = np.asanyarray(frequencies)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3
    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    log_step = np.log(6.4) / 27.0  # step size for log region

    if frequencies.ndim:  # If we have array data, vectorize
        log_t = (frequencies >= min_log_hz)
        mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / log_step

    elif frequencies >= min_log_hz:  # If we have scalar data, heck directly
        mels = min_log_mel + np.log(frequencies / min_log_hz) / log_step

    return mels


def _mel_to_hz(mels):
    """ Convert mel bin numbers to frequencies

    :param mels:    number or np.ndarray [shape=(n,)] -- scalar or array of mel bins to convert

    :return: number or np.ndarray [shape=(n,)] -- input mels in Hz
    """
    # create mels array
    mels = np.asanyarray(mels)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    log_step = np.log(6.4) / 27.0  # step size for log region

    if mels.ndim:  # If we have vector data, vectorize
        log_t = (mels >= min_log_mel)
        freqs[log_t] = min_log_hz * np.exp(log_step * (mels[log_t] - min_log_mel))

    elif mels >= min_log_mel:  # If we have scalar data, check directly
        freqs = min_log_hz * np.exp(log_step * (mels - min_log_mel))

    return freqs


def pre_emphasis_on_mel(mel_spec, preemph, fs, n_mels, mel_fmin=0, mel_fmax=None, min_amplitude=None,
                        normalized_range=True, logger=''):
    logger = get_logger_from_arg(logger)

    # set mel_fmax
    if mel_fmax is None:
        mel_fmax = float(fs) / 2

    #### get the center frequency of all bins in the mel spectrum ####
    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = _hz_to_mel(mel_fmin)
    max_mel = _hz_to_mel(mel_fmax)

    mels = np.linspace(min_mel, max_mel, n_mels + 2)
    bin_freqs = _mel_to_hz(mels)

    #### get the the frequency response of the filter
    a = [1]
    b = [1, -preemph]
    w, h = sps.freqz(b=b, a=a, worN=bin_freqs[1:-1], fs=fs)

    #### apply filter to bins ###
    h = 20 * np.log10(np.abs(h))  # get the filter response in dB
    h = np.tile(np.expand_dims(h, axis=1), (1, mel_spec.shape[1]))

    # if range was normalized
    if normalized_range:
        dbr = -20 * np.log10(min_amplitude)
        # normalize filter
        h = h / dbr

    # Crazy empirical correction hack with magic numbers
    if min_amplitude == 1e-5 and preemph == 0.97:
        correction_matrix = np.log(w) / 30 - 0.277
        correction_matrix = np.tile(np.expand_dims(correction_matrix, axis=1), (1, h.shape[1]))
        h = h - correction_matrix
    else:
        logger.warn("You should probably compute a correction matrix for this config to compensate for the cliping.")

    return np.add(mel_spec, h)
