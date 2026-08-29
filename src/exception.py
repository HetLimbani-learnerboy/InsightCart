"""
Project : InsightCart
File    : exception.py
Purpose : Custom exception handling
"""

import sys


def get_error_message(error, error_detail: sys):

    _, _, exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename

    return (
        f"Error occurred in Python script "
        f"[{file_name}] "
        f"at line [{exc_tb.tb_lineno}] "
        f": {str(error)}"
    )


class CustomException(Exception):

    def __init__(self, error, error_detail: sys):

        super().__init__(str(error))

        self.error_message = get_error_message(error, error_detail)

    def __str__(self):

        return self.error_message
