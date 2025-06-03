import sys
from src.logger import logging

def error_message_details(error, error_details) -> str:
    """Returns a formatted error message with file name and line number."""
    exc_type, exc_value, exc_tb = sys.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
    else:
        file_name = "Unknown"
        line_number = "Unknown"
    error_message = "Error occurred in python script name [{}] line number [{}] error message [{}]".format(
        file_name, line_number, str(error)
    )
    return error_message


class CustomException(Exception):
    """Custom exception class that stores an error message and details."""

    def __init__(self, error_message: str, error_details) -> None:
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_details)
        self.error_details = error_details

    def __str__(self) -> str:
        return self.error_message
    

if __name__ == "__main__":
    try:
        a = 1 / 0  # This will raise a ZeroDivisionError
    except Exception as e:
        logging.info("An error occurred: %s", e)
        raise CustomException(e, sys.exc_info()) from e