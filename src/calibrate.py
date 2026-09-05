"""Compatibility entry point for calibration; training performs validation isotonic fit."""
from .train import train

if __name__ == "__main__":
    train()
