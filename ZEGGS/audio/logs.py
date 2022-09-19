import logging
import os
import sys

from copy import deepcopy
from dateutil.relativedelta import relativedelta


def _get_logger(path_to_file, console_level=logging.DEBUG, file_level=logging.WARNING):
    """ Create a logger object to write in the console and in a file

    :param path_to_file:        name of the log file
    :param console_level:       logging level to write in the console
    :param file_level:          logging level to write in the file

    :return: logger object
    """
    # create folder for log file
    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)

    # create logger
    logger = logging.getLogger(path_to_file)
    logger.setLevel(logging.DEBUG)

    # make it display in the console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    formatter = logging.Formatter('%(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # make it write on file
    f_handler = logging.FileHandler(path_to_file)
    f_handler.setLevel(file_level)
    formatter_f = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    f_handler.setFormatter(formatter_f)
    logger.addHandler(f_handler)

    return logger


def _get_root_logger(console_level=logging.DEBUG):
    """ Create a logger object to write in the console

    :param console_level:   logging level to write in the console

    :return: logger object
    """
    # create logger object
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # make it display in the console
    console_handler = logging.StreamHandler(sys.stdout)

    # check if there is already one stream handler
    already_exist = False
    for l in logger.handlers:
        if type(console_handler) == type(l):
            already_exist = True

    if not already_exist:
        console_handler.setLevel(console_level)
        formatter = logging.Formatter('(root_logger) [%(levelname)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def _format_logging_level_arg(log_level):
    """ Convert logging level arg to its corresponding int logging level

    :param log_level:   arg referring to the desired logging level

    :return: int corresponding to the desired logging level
    """
    if not isinstance(log_level, str):
        if log_level not in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR]:
            print('LOGGING LEVEL NOT RECOGNIZED -- Setting it to DEBUG')
            return logging.DEBUG
        else:
            return log_level

    else:
        if log_level.lower().strip() == 'debug':
            return logging.DEBUG
        elif log_level.lower().strip() == 'info':
            return logging.INFO
        elif log_level.lower().strip() in ['warn', 'warning']:
            return logging.WARNING
        elif log_level.lower().strip() == 'error':
            return logging.ERROR
        else:
            print('LOGGING LEVEL NOT RECOGNIZED -- Setting it to DEBUG')
            return logging.DEBUG


def get_logger_from_arg(logger=None, console_level=logging.DEBUG, file_level=logging.WARNING):
    """ Create logger instance to display information

    :param logger:          either None, string or logger instance
                            can also be a dict of keywords: {'logger': .., console_level: .., file_level: ..}
    :param console_level:   (string or int) logging level to write in the console
    :param file_level:      (string or int) logging level to write in a file

    :return: logger object
    """
    # check if keyword arguments were passed into a dictionary
    if isinstance(logger, dict):
        logger = get_logger_from_arg(**logger)

    # directly check if logger already exists
    elif not isinstance(logger, type(logging.getLogger())) and not isinstance(logger, type(logging.getLogger('dummy'))):
        # convert arguments to logging.level (ints)
        console_level = _format_logging_level_arg(console_level)
        file_level = _format_logging_level_arg(file_level)

        # if the logger is an empty string, return the console print logger
        if logger == '' or logger is None:
            logger = ConsolePrintLogger(console_level)

        # if logger is a string, it is assumed to be the path of the logger object
        elif isinstance(logger, str):
            logger = _get_logger(logger, console_level, file_level)

        # if it is a fake logger continue
        elif isinstance(logger, FakeLogger) or isinstance(logger, ConsolePrintLogger):
            pass

        # default logger
        else:
            logger = ConsolePrintLogger()

    return logger


def get_args_from_logger(logger):
    """ Retrieve args that were given to create logger object

    :param logger:  logger object that was created using get_logger_from_arg()

    :return: dictionary -- {'logger': .., 'console_level': .., 'file_level': ..}
    """
    # set default values
    kwargs = {'logger': None, 'console_level': logging.DEBUG, 'file_level': logging.WARNING}

    # iterate over handlers
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            kwargs['file_level'] = handler.level
            kwargs['logger'] = handler.baseFilename

        elif isinstance(handler, logging.StreamHandler):
            kwargs['console_level'] = handler.level

    return kwargs


def prog_bar(i, n, bar_size=16):
    """ Create a progress bar to estimate remaining time

    :param i:           current iteration
    :param n:           total number of iterations
    :param bar_size:    size of the bar

    :return: a visualisation of the progress bar
    """
    bar = ''
    done = (i * bar_size) // n

    for j in range(bar_size):
        bar += '█' if j <= done else '░'

    message = f'{bar} {i}/{n}'
    return message


def estimate_required_time(nb_items_in_list, current_index, time_elapsed, interval=100):
    """ Compute a remaining time estimation to process all items contained in a list

    :param nb_items_in_list:        all list items that have to be processed
    :param current_index:           current list index, contained in [0, nb_items_in_list - 1]
    :param time_elapsed:            time elapsed to process current_index items in the list
    :param interval:                estimate remaining time when (current_index % interval) == 0

    :return: time elapsed since the last time estimation
    """
    current_index += 1  # increment current_idx by 1
    if current_index % interval == 0 or current_index == nb_items_in_list:
        # make time estimation and put to string format
        seconds = (nb_items_in_list - current_index) * (time_elapsed / current_index)
        time_estimation = relativedelta(seconds=int(seconds))
        time_estimation_string = f'{time_estimation.hours:02}:{time_estimation.minutes:02}:{time_estimation.seconds:02}'

        # extract progress bar
        progress_bar = prog_bar(i=current_index, n=nb_items_in_list)

        # display info
        if current_index == nb_items_in_list:
            sys.stdout.write(f'\r{progress_bar} -- estimated required time = {time_estimation_string} -- Finished!')
        else:
            sys.stdout.write(f'\r{progress_bar} -- estimated required time = {time_estimation_string}')


def simple_table(item_tuples, logger=None):
    """ Display tuple items in a table

    :param item_tuples:     items to display. Each tuple item is composed of two components (heading, cell)
    :param logger:          arg to create logger object
    """
    # get logger object
    logger = get_logger_from_arg(logger)

    # initialize variables
    border_pattern = '+---------------------------------------'
    whitespace = '                                            '

    # extract table items
    headings, cells, = [], []
    for item in item_tuples:
        # extract heading and cell
        heading, cell = str(item[0]), str(item[1])

        # create padding
        pad_head = True if len(heading) < len(cell) else False
        pad = abs(len(heading) - len(cell))
        pad = whitespace[:pad]
        pad_left = pad[:len(pad)//2]
        pad_right = pad[len(pad)//2:]

        if pad_head:  # pad heading
            heading = pad_left + heading + pad_right
        else:  # pad cell
            cell = pad_left + cell + pad_right

        headings += [heading]
        cells += [cell]

    # create the table
    border, head, body = '', '', ''
    for i in range(len(item_tuples)):
        temp_head = f'| {headings[i]} '
        temp_body = f'| {cells[i]} '

        border += border_pattern[:len(temp_head)]
        head += temp_head
        body += temp_body

        if i == len(item_tuples) - 1:
            head += '|'
            body += '|'
            border += '+'

    # display the table
    logger.info(border)
    logger.info(head)
    logger.info(border)
    logger.info(body)
    logger.info(border)
    logger.info(' ')


def get_all_handler_parameters_from_logger(logger):
    """ Extract handler parameters from a logger object

    :param logger:      logger object

    :return: list of handler parameters contained in logger object
    """
    # initialize list
    handler_list = list()

    # iterate over handler parameters
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            handler_list.append({'type': type(h),
                                 'level': h.level,
                                 'file_name': h.baseFilename,
                                 'format': deepcopy(h.formatter),
                                 'object': None})
        else:
            handler_list.append({'type': type(h),
                                 'level': h.level,
                                 'format':deepcopy(h.formatter),
                                 'object': None})

    return handler_list


class FakeLogger:
    """
        FakeLogger is used in multi-processed functions. It replaces a normal logger object.
        It packages message and send them to a queue that a listener will read and write in a proper logger object.
    """
    # class variables
    queue = None
    propagate = False
    level = 0

    def __init__(self, queue):
        self.queue = queue

    def send_fake_message_on_queue(self, level, msg):
        self.queue.put((level, msg))

    def send_warning_of_fake_logger(self, msg):
        msg = f'Logging when using LaFAT multiprocess facilities disables some logger functionality such as: {msg}'
        self.send_fake_message_on_queue(level=logging.WARNING, msg=msg)

    def critical(self, message, *args, **kwargs):
        self.send_fake_message_on_queue(level=logging.CRITICAL, msg=message)

    def error(self, message, *args, **kwargs):
        self.send_fake_message_on_queue(level=logging.ERROR, msg=message)

    def warn(self, message, *args, **kwargs):
        self.send_fake_message_on_queue(level=logging.WARNING, msg=message)

    def warning(self, message, *args, **kwargs):
        self.send_fake_message_on_queue(level=logging.WARNING, msg=message)

    def info(self, message, *args, **kwargs):
        self.send_fake_message_on_queue(level=logging.INFO, msg=message)

    def debug(self, message, *args, **kwargs):
        self.send_fake_message_on_queue(level=logging.DEBUG, msg=message)

    def log(self, message, *args, **kwargs):
        self.send_fake_message_on_queue(level=self.level, msg=message)

    def exception(self, message, *args, **kwargs):
        self.send_fake_message_on_queue(level=logging.ERROR, msg=message)

    def handle(self, record):
        self.queue.put(record)

    def setLevel(self, level):
        self.level = level

    def isEnabledFor(self, level):
        self.send_warning_of_fake_logger(f'isEnabledFor({level})')
        return True

    def getEffectiveLevel(self):
        self.send_warning_of_fake_logger(f'getEffectiveLevel()')
        return 0

    def getChild(self, *args, **kwargs):
        self.send_warning_of_fake_logger(f'getChild()')
        return None

    def addFilter(self, *args, **kwargs):
        self.send_warning_of_fake_logger(f'addFilter()')

    def removeFilter(self, *args, **kwargs):
        self.send_warning_of_fake_logger(f'removeFilter()')

    def filter(self, *args, **kwargs):
        self.send_warning_of_fake_logger(f'filter()')

    def addHandler(self, *args, **kwargs):
        self.send_warning_of_fake_logger(f'addHandler()')

    def removeHandler(self, *args, **kwargs):
        self.send_warning_of_fake_logger(f'removeHandler()')

    def findCaller(self, *args, **kwargs):
        self.send_warning_of_fake_logger(f'findCaller()')

    def makeRecord(self, *args, **kwargs):
        self.send_warning_of_fake_logger(f'makeRecord()')

    def hasHandlers(self):
        return False


class ConsolePrintLogger:
    """
        FakeLogger is used in multi-processed functions. It replaces a normal logger object.
        It packages message and send them to a queue that a listener will read and write in a proper logger object.
    """
    # class variables
    level = 0

    def __init__(self, level=0):
        self.level = level

    def critical(self, message, *args, **kwargs):
        if self.level <= logging.CRITICAL:
            print(f"[CRITICAL]: {message}")

    def error(self, message, *args, **kwargs):
        if self.level <= logging.ERROR:
            print(f"[ERROR]: {message}")

    def warn(self, message, *args, **kwargs):
        if self.level <= logging.WARNING:
            print(f"[WARNING]: {message}")

    def warning(self, message, *args, **kwargs):
        if self.level <= logging.WARNING:
            print(f"[WARNING]: {message}")

    def info(self, message, *args, **kwargs):
        if self.level <= logging.INFO:
            print(f"[INFO]: {message}")

    def debug(self, message, *args, **kwargs):
        if self.level <= logging.DEBUG:
            print(f"[DEBUG]: {message}")

    def log(self, message, *args, **kwargs):
        print(f"[LOG]: {message}")

    def exception(self, message, *args, **kwargs):
        print(f"[EXCEPTION]: {message}")

    def handle(self, record):
        raise Exception("THIS IS MOCK LOGGER, CREATE A REAL ONE IF YOU NEED HANDLES")

    def setLevel(self, level):
        raise Exception("THIS IS MOCK LOGGER, CREATE A REAL ONE IF YOU NEED LEVELS")

    def isEnabledFor(self, level):
        return True

    def getEffectiveLevel(self):
        raise Exception("THIS IS MOCK LOGGER, CREATE A REAL ONE IF YOU NEED LEVELS")
        return 0

    def getChild(self, *args, **kwargs):
        raise Exception("THIS IS MOCK LOGGER, CREATE A REAL ONE CHILDS")
        return None

    def addFilter(self, *args, **kwargs):
        raise Exception("THIS IS MOCK LOGGER, CREATE A REAL ONE IF YOU NEED FILTERS")

    def removeFilter(self, *args, **kwargs):
        raise Exception("THIS IS MOCK LOGGER, CREATE A REAL ONE IF YOU NEED FILTERS")

    def filter(self, *args, **kwargs):
        raise Exception("THIS IS MOCK LOGGER, CREATE A REAL ONE IF YOU NEED FILTERS")

    def addHandler(self, *args, **kwargs):
        raise Exception("THIS IS MOCK LOGGER, CREATE A REAL ONE IF YOU NEED HANDLERS")

    def removeHandler(self, *args, **kwargs):
        raise Exception("THIS IS MOCK LOGGER, CREATE A REAL ONE IF YOU NEED HANDLERS")

    def findCaller(self, *args, **kwargs):
        raise Exception("THIS IS MOCK LOGGER, CREATE A REAL ONE IF YOU NEED CALLERS")

    def makeRecord(self, *args, **kwargs):
        raise Exception("THIS IS MOCK LOGGER, CREATE A REAL ONE IF YOU NEED RECORDS")

    def hasHandlers(self):
        raise Exception("THIS IS MOCK LOGGER, CREATE A REAL ONE IF YOU NEED HANDLERS")
