import datetime, os, sys

# used for output to both console and logfile
class Logger(object):
    def __init__(self, folder_name='logs', abs_path=False, original_terminal=None):
        timestamp = datetime.datetime.now().isoformat()
        self.path = ('./' if not abs_path else '') + folder_name + '/' + str(timestamp).replace(':', '_')
        os.makedirs(self.path)
        self.terminal = sys.stdout if original_terminal is None else original_terminal
        #self.log = open(self.path + "/logfile.log", "a")

    def write(self, message):
        logfile = open(self.path + "/logfile.log", "a")
        self.terminal.write(message)
        logfile.write(message)
        logfile.close()

    def flush(self):
        logfile = open(self.path + "/logfile.log", "a")
        logfile.flush()
        self.terminal.flush()
        logfile.close()

    def get_path(self):
        return self.path