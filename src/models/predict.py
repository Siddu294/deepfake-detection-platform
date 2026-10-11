"""Backward-compatible entry point; use `python -m src.predict <media>` instead."""
from src.predict import main

if __name__ == "__main__":
    main()
