import os
from dotenv import load_dotenv
from pathlib import Path

# Base directory = project root (D:\finance_stock_llm_project)
BASE_DIR = Path(__file__).resolve().parents[1]

# Load environment variables from .env in project root
load_dotenv(BASE_DIR / ".env")

# >>> These exact names must exist for the imports to work <<<
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Symbols for the project (you can change later if you want)
STOCK_SYMBOLS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]

# Data directories
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Date ranges for training/testing
TRAIN_START = "2015-01-01"
TRAIN_END = "2021-12-31"
TEST_START = "2022-01-01"
TEST_END = "2024-12-31"
