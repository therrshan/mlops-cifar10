import logging
import os
from datetime import datetime

class CustomLogger:
    def __init__(self, log_dir="../logs", log_filename=None, log_level=logging.INFO):
        """Initializes the logger with console and file handlers."""
        os.makedirs(log_dir, exist_ok=True)  # Ensure log directory exists

        if log_filename is None:
            log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
        log_filepath = os.path.join(log_dir, log_filename)

        # Create logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        # Prevent duplicate handlers
        if not self.logger.hasHandlers():
            # File Handler
            file_handler = logging.FileHandler(log_filepath)
            file_handler.setFormatter(self._log_formatter())

            # Console Handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(self._log_formatter())

            # Add handlers
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

            self.logger.info(f"Logger initialized. Logs saved to {log_filepath}")

    def _log_formatter(self):
        """Returns a log formatter with timestamp, log level, and message."""
        return logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s", "%Y-%m-%d %H:%M:%S")

    def get_logger(self):
        """Returns the logger instance."""
        return self.logger


# Example usage
if __name__ == "__main__":
    logger = CustomLogger().get_logger()
    logger.info("This is an INFO message.")
    logger.debug("This is a DEBUG message.")
    logger.warning("This is a WARNING message.")
    logger.error("This is an ERROR message.")
    logger.critical("This is a CRITICAL message.")
