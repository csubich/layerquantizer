import os
import sys

# Add parent directory to sys.path to allow importing from utils.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import clear_cache

if __name__ == "__main__":
    clear_cache()
