import argparse
import subprocess
import sys
from app_logger import setup_logger

def cli():
    """
    Agentic RAG Chatbot (Azure Native) CLI entry point.
    Provides options to run different modules: frontend, backend, auth.
    """

    # Initialize logger
    logger, log_filename = setup_logger(name="agentic-rag")
    logger.info("Agentic RAG Chatbot CLI started")
    logger.info(f"Logs are being written to {log_filename}")

    parser = argparse.ArgumentParser(
        prog="agentic-rag",
        description="Agentic RAG Chatbot (Azure Native) CLI"
    )

    parser.add_argument(
        "--frontend", action="store_true",
        help="Run the frontend (Chainlit UI)"
    )
    parser.add_argument(
        "--backend", action="store_true",
        help="Run the backend service (FastAPI or similar)"
    )
    parser.add_argument(
        "--auth", action="store_true",
        help="Run the authentication service"
    )

    args = parser.parse_args()

    try:
        if args.frontend:
            logger.info("Starting frontend (Chainlit UI)...")
            subprocess.run([sys.executable, "-m", "chainlit", "run", "src/frontend/app.py"])
        elif args.backend:
            logger.info("Starting backend service...")
            subprocess.run([sys.executable, "src/backend/app.py"])
        elif args.auth:
            logger.info("Starting auth service...")
            subprocess.run([sys.executable, "src/auth/app.py"])
        else:
            parser.print_help()
    except Exception as e:
        logger.error(f"An error occurred while running CLI: {e}")
        sys.exit(1)