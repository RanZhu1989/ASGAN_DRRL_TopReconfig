import os
import torch
from datetime import datetime

#---------------------------------------------------------------------------------------
"""
loggers
"""
import sys
import os
import logging
import csv
from datetime import datetime



class event_logger:

    def __init__(self,
                 log_output_path,
                 user_label = "default",
                 create_file = False,
                 level = logging.DEBUG,
                 verbose = 0):
        """
        .log file
        """
        self.log_output_path = log_output_path
        self.level = level
        self.verbose = verbose
        if create_file:
            self.reset(log_output_path=log_output_path,
                       user_label=user_label,
                       level=level,
                       verbose=verbose)
        pass

    def reset(self,
              log_output_path,
              user_label = "default",
              level = logging.DEBUG,
              verbose = 0):
        """
        Method to reset the logger
        """
        now = datetime.now()
        dt_string = now.strftime("__%Y_%m_%d_%H_%M")

        if not os.path.isdir(log_output_path):
            os.makedirs(log_output_path)
        # Logger
        event_log_output_path = str(log_output_path /
                                    (user_label + "_log" + dt_string + ".log"))

        # create the logger
        self.logger = logging.getLogger(event_log_output_path)
        self.logger.setLevel(level)
        format_string = (
            "%(asctime)s - %(levelname)s - %(funcName)s (%(lineno)d):  %(message)s"
        )
        datefmt = "%Y-%m-%d %I:%M:%S %p"
        log_format = logging.Formatter(format_string, datefmt)

        # Creating and adding the file handler
        file_handler = logging.FileHandler(event_log_output_path, mode="a")
        file_handler.setFormatter(log_format)
        self.logger.addHandler(file_handler)

        if verbose == 1:
            # Creating and adding the console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(log_format)
            self.logger.addHandler(console_handler)

        pass

    def reset_user_label(self, user_label):
        self.reset(self.log_output_path, user_label, self.level, self.verbose)
        pass


class data_logger:

    def __init__(self,
                 log_output_path,
                 user_label = "default") :
        """
        .csv file
        """
        self.log_output_path = log_output_path
        self.reset(log_output_path, user_label)
        pass

    def reset(self,
              log_output_path,
              user_label = "default",
              reset_time = True):
        now = datetime.now()
        if reset_time:
            self.dt_string = now.strftime("__%Y_%m_%d_%H_%M")
        # variables
        self.log_path = str(log_output_path /
                            (user_label + self.dt_string + ".csv"))
        pass

    def reset_user_label(self, user_label):
        self.reset(self.log_output_path, user_label, reset_time=False)
        pass

    def save_to_csv(self, log):
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w"):
                pass
        with open(self.log_path, "a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(log)

        pass

    #---------------------------------------------------------------------------------------


def get_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def check_cuda():
    # check cuda
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        # print('use_gpu ==', use_gpu)
        # print('device_ids ==', np.arange(0, torch.cuda.device_count()))
        return torch.device('cuda')
    else:
        return torch.device('cpu')
