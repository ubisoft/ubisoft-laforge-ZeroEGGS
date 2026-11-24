import os
from math import gcd

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from . import get_logger_from_arg


def read_wavfile(file_path, rescale=False, desired_fs=None, desired_nb_channels=None, out_type='float32', logger=None):
    """ Read a WAV file and return the samples in a float32 numpy array

    :param file_path:               path to the file to read
    :param rescale:                 rescale the file to get amplitudes in the range between -1 and +1
                                    only the range is rescaled, not the amplitude
    :param desired_fs:              frequency expected from the WAV file
                                    if not specified, the original WAV file sampling frequency is used
    :param desired_nb_channels:     number of channels expected from the WAV file
                                    if not specified, the original WAV number of channels is used
    :param out_type:            desired output type of the audio waveform
    :param logger:                  arg to create logger object

    :return: sampling frequency and samples
    """
    # create logger object
    logger = get_logger_from_arg(logger)

    # check arguments make sense
    assert ('int' in out_type or 'float' in out_type), \
        logger.error(f'Inconsistent argument: only output of type "int" or "float" are supported, not "{out_type}"')
    if rescale:
        assert ('float' in out_type), logger.error(f'Inconsistent arguments: cannot rescale if out_type={out_type}')

    # normalize and strip path
    file_path = os.path.normpath(file_path).strip()

    fs, x = wavfile.read(file_path)
    nb_channels = 1 if x.ndim == 1 else x.shape[1]
    if (desired_fs and fs != desired_fs) or (desired_nb_channels and nb_channels != desired_nb_channels):
        # Downmix to the requested channel count (mono by default) and resample in memory.
        desired_fs = desired_fs or fs
        desired_nb_channels = desired_nb_channels or 1
        logger.info(f'{file_path}: converting {fs}Hz/{nb_channels}ch to {desired_fs}Hz/{desired_nb_channels}ch')
        x = _rescale_wav_to_float32(x).astype(np.float64)
        if x.ndim > 1 and desired_nb_channels == 1:
            x = x.mean(axis=1)
        g = gcd(int(fs), int(desired_fs))
        x = np.clip(resample_poly(x, desired_fs // g, fs // g, axis=0), -1.0, 1.0)
        x = _rescale_wav_from_float32(x.astype(np.float32), out_type)
        fs = desired_fs

    # rescale between -1 and 1 in float32
    if rescale:
        x = _rescale_wav_to_float32(x)

    # extract current waveform dtype and check everything is correct
    current_dtype = str(x.dtype)
    if 'int' in current_dtype and 'float' in out_type:
        logger.warning(f'Waveform is "{current_dtype}", converting to "{out_type}" but values will not be in '
                       f'[-1., 1.] -- Use rescale=True to have samples in [-1., 1.]')
    if 'float' in current_dtype:  # sample values are in [-1., 1.]
        assert ('int' not in out_type), logger.error(f'Waveform is "{current_dtype}", cannot convert to "{out_type}"')

    # cast to desired output type
    x = np.asarray(x).astype(out_type)

    return fs, x


def write_wavefile(fileName, pcmData, sampling_rate, out_type='int16'):
    """ write a WAV file from a numpy array

    :param fileName:                path and file name to write to
    :param pcmData:                 The numpy array containing the PCM data
    :param sampling_rate:           the sampling rate of the data
    :param out_type:                desired output type of the audio waveform
    """
    current_dtype = str(pcmData.dtype)
    if 'float' in current_dtype and out_type == 'int16':  # sample values are in [-1., 1.] convert to [-32k, 32k]
        data = pcmData * 2 ** 15
    else:
        data = pcmData

    data = data.astype(out_type)
    wavfile.write(fileName, sampling_rate, data)


def _rescale_wav_to_float32(x):
    """ Rescale WAV array between -1.f and 1.f based on the current format

    :param x:           audio array

    :return: the rescaled audio array in float32
    """

    # rescale audio array
    y = np.zeros(x.shape, dtype='float32')
    if x.dtype == 'int16':
        y = x / 32768.0
    elif x.dtype == 'int32':
        y = x / 2147483648.0
    elif x.dtype == 'float32' or x.dtype == 'float64':
        max_ampl = np.max(np.abs(x))
        if max_ampl > 1.0:
            raise ValueError(f'float32 wav contains samples not in the range [-1., 1.] -- '
                             f'max amplitude: {max_ampl}')
        y = x.astype('float32')
    elif x.dtype == 'uint8':
        y = ((x / 255.0) - 0.5) * 2
    else:
        raise TypeError(f"could not normalize wav, unsupported sample type {x.dtype}")

    return y


def _rescale_wav_from_float32(x, dtype):
    """ Rescale WAV array from between -1.f and 1.f to the provided format/dtype

    :param x:           audio array
    :param dtype:       numpy dtype to scale to

    :return: the rescaled audio array in specified format/dtype
    """

    max_ampl = np.max(np.abs(x))
    if max_ampl > 1.0:
        raise ValueError(f'float32 wav contains samples not in the range [-1., 1.] -- ' \
                         f'max amplitude: {max_ampl}')

    # rescale audio array
    y = np.zeros(x.shape, dtype=dtype)
    if dtype == 'int16':
        y = x * 32768.0
    elif dtype == 'int32':
        y = x * 2147483648.0
    elif dtype == 'float32' or dtype == 'float64':
        y = x
    elif dtype == 'uint8':
        y = 255.0 * ((x / 2.0) + 0.5)
    else:
        raise TypeError(f"could not normalize wav, unsupported sample type {x.dtype}")

    # convert numpy array to provided type
    y = y.astype(dtype)

    return y

