#!/usr/bin/env python3
"""Download and setup all required models and data for the ATT&CK Extractor"""
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

import requests
from tqdm import tqdm
from transformers import AutoModel, AutoModelForTokenClassification, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelDownloader:
    def __init__(self):
        self.models_dir = Path("models")
        self.data_dir = Path("data")

        # Create directories
        self.models_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)

        # List of models to download
        self.models = {
            "secureBERT": {
                "path": self.models_dir / "secureBERT",
                "source": "ehsanaghaei/SecureBERT",
                "type": "transformers",
            },
            "secureBERT-NER": {
                "path": self.models_dir / "SecureBERT-NER",
                "source": "CyberPeace-Institute/SecureBERT-NER",
                "type": "transformers",
            },
            "bge-large-en-v1.5": {
                "path": self.models_dir / "bge-large-en-v1.5",
                "source": "BAAI/bge-large-en-v1.5",
                "type": "transformers",
            },
        }

        # Data files to verify
        self.data_files = {
            "CAPEC.csv": "CAPEC attack patterns",
            "d3fend-full-mappings.csv": "D3FEND mappings",
            "Engage-Data-V1.0.csv": "MITRE Engage data",
            "nvd_vulnerabilities_merged.csv": "NVD vulnerabilities",
            "kev_cve_cwe.csv": "KEV CVE-CWE mappings",
            "kev_cve_attack.csv": "KEV CVE-ATT&CK mappings",
            "technique_keywords.json": "ATT&CK technique keywords",
        }

    def download_model(self, name: str, model_info: dict):
        """Download a transformers model from HuggingFace"""
        logger.info(f"Downloading {name}...")

        if model_info["path"].exists():
            logger.info(f"{name} already exists, skipping...")
            return

        try:
            # Download and save the model
            model = AutoModel.from_pretrained(model_info["source"])
            tokenizer = AutoTokenizer.from_pretrained(model_info["source"])

            # Save to local directory
            model.save_pretrained(model_info["path"])
            tokenizer.save_pretrained(model_info["path"])

            logger.info(f"Successfully downloaded {name}")
        except Exception as e:
            logger.error(f"Failed to download {name}: {e}")
            if model_info["path"].exists():
                shutil.rmtree(model_info["path"])

    def download_all_models(self):
        """Download all required models"""
        for name, info in self.models.items():
            self.download_model(name, info)

    def verify_data_files(self):
        """Verify that all required data files are present"""
        missing_files = []

        for filename, description in self.data_files.items():
            filepath = self.data_dir / filename
            if not filepath.exists():
                missing_files.append((filename, description))
                logger.warning(f"Missing data file: {filename} ({description})")
            else:
                logger.info(f"Found: {filename}")

        if missing_files:
            logger.error(
                "Missing data files. Please ensure all data files are uploaded to the data/ directory"
            )
            for filename, desc in missing_files:
                logger.error(f"  - {filename}: {desc}")
            return False
        else:
            logger.info("All data files verified successfully")
            return True

    def create_summary(self):
        """Create a summary of downloaded resources"""
        summary = {
            "models": {},
            "data_files": {},
            "timestamp": str(Path("models/.download_complete").stat().st_mtime),
        }

        # Check models
        for name, info in self.models.items():
            summary["models"][name] = {
                "exists": info["path"].exists(),
                "size": (
                    sum(
                        f.stat().st_size for f in info["path"].rglob("*") if f.is_file()
                    )
                    if info["path"].exists()
                    else 0
                ),
            }

        # Check data files
        for filename in self.data_files:
            filepath = self.data_dir / filename
            summary["data_files"][filename] = {
                "exists": filepath.exists(),
                "size": filepath.stat().st_size if filepath.exists() else 0,
            }

        # Save summary
        with open(self.models_dir / "download_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("Download summary saved to models/download_summary.json")


def main():
    downloader = ModelDownloader()

    logger.info("Starting model and data setup...")

    # Download all models
    downloader.download_all_models()

    # Verify data files
    data_verified = downloader.verify_data_files()

    # Create download complete marker
    if data_verified:
        (downloader.models_dir / ".download_complete").touch()
        logger.info("Download and verification complete!")

        # Create summary
        downloader.create_summary()
    else:
        logger.error("Download complete but data verification failed")

    logger.info(
        f"Total size: {sum(f.stat().st_size for f in downloader.models_dir.rglob('*') if f.is_file()) / (1024*1024):.1f} MB"
    )
    logger.info("Setup completed successfully!")


if __name__ == "__main__":
    main()
