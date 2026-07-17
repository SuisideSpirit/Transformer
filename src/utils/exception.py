import sys

class TransformerException(Exception):
    def __init__(self, error_message, error_detail: sys):
        self.error_message = self.get_detailed_error_message(
            error_message,
            error_detail,
        )
        super().__init__(self.error_message)

    @staticmethod
    def get_detailed_error_message(error_message, error_detail: sys):
        type, value, exc_tb = error_detail.exc_info()

        if exc_tb is None:
            return error_message

        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno

        return (
            f"Type of error {type}"
            f"Value of error {value}"
            f"Error occurred in script [{file_name}] "
            f"at line [{line_number}] : {error_message}"
        )

    def __str__(self):
        return self.error_message