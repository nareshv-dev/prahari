"""Download IEEE-CIS data without ever committing raw files."""
from pathlib import Path
import os

def download():
    out = Path(__file__).parent / "raw"; out.mkdir(parents=True, exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
        api.competition_download_files("ieee-fraud-detection", path=str(out))
        import zipfile
        z = next(out.glob("*.zip")); zipfile.ZipFile(z).extractall(out)
        print(f"Downloaded IEEE-CIS data to {out}")
    except Exception as exc:
        raise RuntimeError("Kaggle credentials missing or download failed. Set KAGGLE_USERNAME/KAGGLE_KEY "
                           "or ~/.kaggle/kaggle.json, then run `python data/download.py`. "
                           f"Details: {exc}") from exc

if __name__ == "__main__": download()