import os
import argparse
from util.args import save_args
from datetime import datetime


class Log:

    """
    Object for managing the log directory
    """

    def __init__(self, log_dir: str, mode: str = 'a'):

        self._log_dir = log_dir
        self._logs = dict()
        self._mode = mode

        # Ensure the directories exist
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        open(self.log_dir + '/log.txt', self._mode).close()

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def checkpoint_dir(self):
        return self._log_dir + '/checkpoints'

    @property
    def metadata_dir(self):
        return self._log_dir + '/metadata'

    @staticmethod
    def timestamp() -> str:
        now = datetime.now()
        return now.strftime("[%d/%m/%Y %H:%M:%S] ")

    def log_message(self, msg: str):
        """
        Write a message to the log file
        :param msg: the message string to be written to the log file
        """
        with open(self.log_dir + '/log.txt', 'a') as f:
            timestamp = self.timestamp()
            # Move \n if necessary
            msg = '\n'+timestamp+msg[1:] if msg.startswith('\n') else timestamp+msg
            f.write(msg+"\n")

    def create_log(self, log_name: str, key_name: str, *value_names):
        """
        Create a csv for logging information
        :param log_name: The name of the log. The log filename will be <log_name>.csv.
        :param key_name: The name of the attribute that is used as key (e.g. epoch number)
        :param value_names: The names of the attributes that are logged
        """
        if log_name in self._logs.keys():
            raise Exception('Log already exists!')
        # Add to existing logs
        self._logs[log_name] = (key_name, value_names)
        # Create log file. Create columns
        with open(self.log_dir + f'/{log_name}.csv', self._mode) as f:
            f.write(','.join((key_name,) + value_names) + '\n')

    def log_values(self, log_name, key, *values):
        """
        Log values in an existent log file
        :param log_name: The name of the log file
        :param key: The key attribute for logging these values
        :param values: value attributes that will be stored in the log
        """
        if log_name not in self._logs.keys():
            raise Exception('Log not existent!')
        if len(values) != len(self._logs[log_name][1]):
            raise Exception('Not all required values are logged!')
        # Write a new line with the given values
        with open(self.log_dir + f'/{log_name}.csv', 'a') as f:
            f.write(','.join(str(v) for v in (key,) + values) + '\n')

    def log_args(self, args: argparse.Namespace):
        save_args(args, self._log_dir)
