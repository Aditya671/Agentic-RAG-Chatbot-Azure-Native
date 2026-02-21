
import logging
import os
from datetime import datetime
from sys import stdout

def setup_logger(name="my-app", log_dir='./logs', log_level=logging.DEBUG):
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Log file name with timestamp
    log_filename = datetime.now().strftime(f"{name}_%Y-%m-%d_%H-%M-%S.log")
    log_path = os.path.join(log_dir, log_filename)

    # Logger setup
    logger = logging.getLogger(name)
    
    logger.setLevel(log_level)

    # Formatter
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    logger.handlers = []
    # File handler
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    if not file_handler in logger.handlers:
        logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(stdout)
    console_handler.setFormatter(formatter)
    if not console_handler in logger.handlers:
        logger.addHandler(console_handler)
    # Additional loggers for third-party libraries
    logging.getLogger("llama_index").setLevel(logging.INFO)
    logging.getLogger("llama_index.agent").setLevel(logging.INFO)
    logging.getLogger("llama_index.vector_stores").setLevel(logging.WARNING)
    logging.getLogger("llama_index.storage").setLevel(logging.WARNING)
    logging.getLogger("llama_index.agent.function_calling").setLevel(logging.INFO)
    logging.getLogger("llama_index.tools.retriever").setLevel(logging.INFO)

    # Suppress Azure SDK info/debug logs
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    from azure.storage.blob import BlobServiceClient
    from azure.core.pipeline.policies import HttpLoggingPolicy
    logging.getLogger('azure').setLevel(logging.WARNING)

    # Disable logging of request/response bodies
    http_logging_policy = HttpLoggingPolicy()
    http_logging_policy.allowed_header_names = set()  # Remove headers to log
    http_logging_policy.allowed_query_params = set()  # Remove query params to log

    # Optional: Suppress HTTP requests logs (urllib3, requests, etc.)
    logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.WARNING)

    return logger, log_filename

# logger, log_filename = setup_logger()
def filter_error_logs(input_file, output_file_name = "Error_log", logger = logging.getLogger()):
    # Define the keyword you are looking for
    search_keyword = "ERROR"
    output_file = f"""{datetime.now().strftime(f"{output_file_name}_%Y-%m-%d_%H-%M-%S.log")}"""
    try:
        with open(f"../logs/{input_file}", 'r', encoding='utf-8', errors="ignore") as infile, open(f"../logs/{output_file}", 'w') as outfile:
            for line in infile:
                # Check if 'ERROR' is in the line (case-insensitive)
                if search_keyword in line.upper():
                    outfile.write(line)
        
        logger.info(f"Filtering complete. Errors saved to {output_file}")
        
    except FileNotFoundError:
        logger.error("The source log file was not found.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

# filter_error_logs('system.log', 'errors_only.log', logger)