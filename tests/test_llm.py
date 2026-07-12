import sys
import os

# Add src to path
sys.path.append(os.path.abspath('src'))

from src.llm_client import test_llm_connection
from src.config import CHAT_MODEL

def test():
    print(f"Testing connection with model: {CHAT_MODEL}")
    success, message, error = test_llm_connection()
    if success:
        print(f"SUCCESS: {message}")
    else:
        print(f"FAILURE: {error}")

if __name__ == "__main__":
    test()
