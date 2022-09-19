import os

import numpy as np
import sox
from scipy.io import wavfile

from .logs import get_logger_from_arg


def reformat_and_trim_wav_file(wav_file, fs, bit_depth, nb_channels, overwrite=True, out_path=None,
                               silence_threshold=0.1, min_silence_duration=0.01, silence_pad=True, logger=None):
    """ Format WAV files with the specified parameters using SoX

    :param wav_file:                WAV file to format (full path)
    :param fs:                      desired sampling frequency of WAV file
    :param bit_depth:               desired bit depth of WAV file
    :param nb_channels:             desired number of channels of WAV file
    :param overwrite:               overwrite existing WAV file with their new version
                                    if not, a folder is created to store the new files
    :param out_path:                path to save reformatted WAV file
                                    only used when overwrite is set to False
    :param silence_threshold:       threshold to detect silences
    :param min_silence_duration:    min silence duration to remove
                                    only used when silence_threshold is superior to 0.
    :param silence_pad:             pad audio with silences at the beginning and the end
    :param logger:                  arg to create logger object
    """
    # create logger object
    logger = get_logger_from_arg(logger)

    # ---------- DEAL WITH PATHS ----------

    # normalize and strip path
    initial_path = os.path.normpath(wav_file).strip()

    if overwrite:
        # create a temporary filename
        out_path = os.path.join(os.path.dirname(initial_path),
                                os.path.basename(initial_path).replace('.wav', '_tmp.wav'))
    else:
        if out_path:
            # processed WAV file name
            out_path = os.path.normpath(out_path).strip()
        else:
            # create a folder named processed at the file location
            out_path = os.path.join(os.path.dirname(initial_path), f'processed_{fs}')
            os.makedirs(out_path, exist_ok=True)
            out_path = os.path.join(out_path, os.path.basename(initial_path))

    # ---------- REFORMAT FILE WITH SOX ----------

    # create transformer
    tfm = sox.Transformer()

    # remove silences
    if silence_threshold > 0.:
        # remove silence at the beginning
        tfm.silence(location=1, silence_threshold=silence_threshold,
                    min_silence_duration=min_silence_duration, buffer_around_silence=True)
        # remove silence at the end
        tfm.silence(location=-1, silence_threshold=silence_threshold,
                    min_silence_duration=min_silence_duration, buffer_around_silence=True)

    # re-sample to desired frequency
    tfm.rate(samplerate=fs, quality='h')

    # convert to desired bit depth and number of channels
    tfm.convert(samplerate=None, n_channels=nb_channels, bitdepth=bit_depth)

    # add short silences at the end and beginning of the file
    if silence_pad:
        tfm.pad(start_duration=0.01, end_duration=0.01)

    # display the applied effects in the logger
    logger.info(f'SoX transformer effects: {tfm.effects_log}')

    # create the output file.
    tfm.build(initial_path, out_path)

    # ---------- CLEAN-UP ----------

    # delete original file and replace by new file
    if overwrite:
        os.remove(initial_path)
        os.rename(out_path, initial_path)


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

    try:
        # try to read the wav file
        fs, x = wavfile.read(file_path)

        # raise exception if sampling frequency, bit depth or number of channels are not correct
        current_bit_depth = int(str(x.dtype).replace('int', '').replace('uint', '').replace('float', ''))
        if desired_fs and fs != desired_fs or desired_nb_channels and len(x.shape) != desired_nb_channels:
            raise BadSamplingFrequencyError(f'Format readable but requirements not met -- currently is '
                                            f'{fs}Hz/{current_bit_depth} bits/{len(x.shape)} channels')

    except (ValueError, BadSamplingFrequencyError) as e:
        # create a reformatted temporary version
        tmp_wav = os.path.join(os.path.dirname(file_path),
                               os.path.basename(file_path).replace('.wav', '_tmp.wav'))

        # add default value if nothing is specified
        desired_fs = desired_fs if desired_fs else 22050
        desired_nb_channels = desired_nb_channels if desired_nb_channels else 1

        # infer desired bit depth with desired out_type
        desired_bit_depth = int(out_type.replace('int', '').replace('uint', '').replace('float', ''))

        # reformat
        logger.info(f'{file_path} -- {e}')
        logger.info(f'converting to {desired_fs}Hz/{desired_bit_depth} bits/{desired_nb_channels} channels')
        reformat_and_trim_wav_file(file_path, fs=desired_fs, bit_depth=desired_bit_depth,
                                   nb_channels=desired_nb_channels, overwrite=False, out_path=tmp_wav,
                                   silence_threshold=-1., silence_pad=False, logger=logger)

        # read reformatted file and delete it
        fs, x = wavfile.read(tmp_wav)
        os.remove(tmp_wav)

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


def rescale_wav_array(x, desired_dtype='float32'):
    """ Rescale WAV array to a specified dtype

    rescales the samples in the given array from the range of its current dtype
    to the range of the specified dtype.  see ranges by type below...

    float32 samples are assumed to be in the range [-1.0,1.0], otherwise an exception is raised.

    =====================  ===========  ===========  =============
         WAV format            Min          Max       NumPy dtype
    =====================  ===========  ===========  =============
    32-bit floating-point  -1.0         +1.0         float32
    32-bit PCM             -2147483648  +2147483647  int32
    16-bit PCM             -32768       +32767       int16
    8-bit PCM              0            255          uint8
    =====================  ===========  ===========  =============

    :param x:               audio array
    :param desired_dtype:   nuympy dtype to rescale to

    :return: the rescaled audio array in float32
    """
    y = _rescale_wav_to_float32(x)
    z = _rescale_wav_from_float32(y, desired_dtype)
    return z


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


class BadSamplingFrequencyError(Exception):
    def __init__(self, message):
        self.message = message
