import sys   # sys library provides various functions and variables that are used to manipulate different parts of the Python runtime environment.  
import logging  # We are importing the logging module to log the error messages. This will help us to keep track of the errors and also to debug the code. We can log the error messages in a file or in the console depending on our requirement. In this project, we will log the error messages in a file. We will create a separate file for logging the error messages and we will use the logging module to log the error messages in that file. We will also use the logging module to log the error messages in the console for debugging purposes.


def error_message_detail(error, error_detail:sys):
    _, _, exc_tb = error_detail.exc_info()  # exc_info() is a function in the sys module that returns a tuple containing information about the exception that is currently being handled. The tuple contains three values: the exception type, the exception value, and the traceback object.
    file_name = exc_tb.tb_frame.f_code.co_filename  # The traceback object has an attribute called tb_frame, which is a frame object representing the execution frame where the exception occurred. The frame object has an attribute called f_code, which is a code object representing the compiled bytecode of the function or module where the exception occurred. The code object has an attribute called co_filename, which is a string representing the filename of the source code where the exception occurred.
    error_message = f"Error occurred in script: [{file_name}] line number: [{exc_tb.tb_lineno}] error message: [{str(error)}]"  # The traceback object also has an attribute called tb_lineno, which is an integer representing the line number in the source code where the exception occurred. We are using this information to create a detailed error message that includes the filename, line number, and error message.
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)  # The super()(inheriting) function is used to call the __init__ method of the parent class (Exception) and pass the error_message to it. This allows us to create a custom exception class that inherits from the built-in Exception class and can be used to raise exceptions with detailed error messages.
        self.error_message = error_message_detail(error_message, error_detail=error_detail)  # We are calling the error_message_detail function to get the detailed error message and storing it in an instance variable called error_message.

    def __str__(self):
        return self.error_message  # The __str__ method is a special method in Python that is called when we try to convert an object to a string. In this case, we are overriding the __str__ method to return the detailed error message when we print the CustomException object.
    

# # Initial test
# if __name__ == "__main__":

#     try:
#         a = 1 / 0  # This will raise a ZeroDivisionError because we cannot divide a number by zero.
#     except Exception as e:
#         logging.info("An error occurred.")  # We are logging an info message to indicate that an error has occurred. This is just a test to check if the logging is working properly. We can remove this line later when we start using the logging in our project.
#         raise CustomException(e, sys)  # We are raising a CustomException with the original exception and the sys module as arguments. This will allow us to get the detailed error message when the exception is raised.
    
