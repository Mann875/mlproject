import logging # Any execution that happens we will be able to log all information into some files. applies to custom exceptions, errors etc.  
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m-%d-%Y_%H_%M_%S')}.log"  
# We are creating a log file with the current date and time as the name. This will help us to keep track of the logs and also to avoid overwriting the logs.
logs_path = os.path.join(os.getcwd(), 'logs')  
# We are creating a path for the log file. We are using os.getcwd() to get the current working directory and then joining it with the 'logs' folder and the log file name.
os.makedirs(logs_path, exist_ok=True)
# We are creating the logs folder if it does not exist. The os.makedirs() function is used to create a directory recursively. The os.path.dirname() function is used to get the directory name from the logs_path. The exist_ok=True parameter is used to avoid raising an error if the directory already exists.   

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)
# We are creating the full path for the log file by joining the logs_path with the log file name.
logging.basicConfig(
    filename=LOG_FILE_PATH,  
    # We are specifying the filename for the log file.     
    format='[%(asctime)s] %(lineno)d%(name)s - %(levelname)s - %(message)s',  
    # We are specifying the format for the log messages. The format includes the timestamp, logger name, log level and the log message.
    level=logging.INFO,  
    # We are setting the log level to INFO. This means that all log messages with a level of INFO or higher will be logged. The log levels are DEBUG, INFO, WARNING, ERROR and CRITICAL. By setting the log level to INFO, we will be able to log all the important information without logging too much debug information.
)



# # Initial test
# if __name__ == "__main__":
#     logging.info("Logging has started.")  # We are logging an info message to indicate that the logging has started. This is just a test to check if the logging is working properly. We can remove this line later when we start using the logging in our project. 

