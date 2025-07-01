"""
KEV Mapper for MITRE ATT&CK Integration
---------------------------------------
Maps Known Exploited Vulnerabilities (KEV) to ATT&CK techniques.
Uses only existing data from CSV files without generating additional mappings.
"""

import csv
import json
import logging
import os
import re
from typing import Dict, List, Optional, Set, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)


class KEVMapper:
    """
    Maps KEV data to ATT&CK techniques using predefined mappings from CSV files
    """

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the KEV mapper

        Args:
            data_dir: Directory containing KEV data files
        """
        self.data_dir = data_dir
        self.kev_cve_cwe_path = os.path.join(data_dir, "kev_cve_cwe.csv")
        self.kev_cve_attack_path = os.path.join(data_dir, "kev_cve_attack.csv")

        # Data stores
        self.kev_data = {}  # CVE -> KEV data
        self.cve_technique_map = {}  # CVE -> Techniques
        self.cve_cwe_map = {}  # CVE -> CWEs
        self.cwe_technique_map = (
            {}
        )  # CWE -> Techniques (will remain empty, not generated)

        logger.info(f"KEV mapper initialized with data directory: {data_dir}")

    def load_kev_data(self) -> bool:
        """
        Load KEV data from the kev_cve_cwe.csv file

        Returns:
            Whether loading was successful
        """
        if not os.path.exists(self.kev_cve_cwe_path):
            logger.warning(f"KEV CVE-CWE CSV not found at {self.kev_cve_cwe_path}")
            return False

        try:
            # Load KEV CSV using pandas for better column handling
            df = pd.read_csv(self.kev_cve_cwe_path)
            logger.info(
                f"Successfully read {len(df)} rows from {self.kev_cve_cwe_path}"
            )

            # Convert to dictionary
            self.kev_data = {}
            self.cve_cwe_map = {}

            for _, row in df.iterrows():
                # Get CVE ID - ensure it exists and is properly formatted
                cve_id = row.get("cveID")

                if not cve_id or not isinstance(cve_id, str):
                    continue

                # Ensure CVE-ID format
                if not cve_id.startswith("CVE-"):
                    cve_id = f"CVE-{cve_id}"

                # Extract CWEs if available
                cwes = []
                if pd.notna(row.get("cwes")):
                    # Handle potential list format or string format
                    cwe_value = row.get("cwes")
                    if isinstance(cwe_value, str):
                        # Extract CWE IDs using regex
                        cwe_matches = re.findall(r"CWE-\d+", cwe_value)
                        if cwe_matches:
                            cwes = cwe_matches
                        # Split by commas if multiple CWEs and no matches found
                        elif "," in cwe_value:
                            # Process each potential CWE ID
                            for cwe_part in cwe_value.split(","):
                                cwe_part = cwe_part.strip()
                                if cwe_part.isdigit():
                                    cwes.append(f"CWE-{cwe_part}")
                                elif cwe_part.startswith("CWE-"):
                                    cwes.append(cwe_part)
                        elif cwe_value.isdigit():
                            cwes = [f"CWE-{cwe_value}"]
                        elif cwe_value.startswith("CWE-"):
                            cwes = [cwe_value]

                # Store in CWE map
                if cwes:
                    self.cve_cwe_map[cve_id] = cwes

                # Create KEV data entry with all available columns
                entry = {
                    "cwes": cwes,
                    "vendorProject": "",
                    "product": "",
                    "vulnerabilityName": "",
                    "dateAdded": "",
                    "shortDescription": "",
                    "requiredAction": "",
                    "dueDate": "",
                    "notes": "",
                }

                # Add all available fields from the row
                for field in [
                    "vendorProject",
                    "product",
                    "vulnerabilityName",
                    "dateAdded",
                    "shortDescription",
                    "requiredAction",
                    "dueDate",
                    "notes",
                ]:
                    if field in row and pd.notna(row[field]):
                        entry[field] = row[field]

                # Store the entry
                self.kev_data[cve_id] = entry

            logger.info(f"Loaded {len(self.kev_data)} entries from KEV CVE-CWE data")
            logger.info(f"Mapped {len(self.cve_cwe_map)} CVEs to CWEs")
            return True

        except Exception as e:
            logger.error(f"Error loading KEV data: {e}")
            return False

    def load_cve_attack_mappings(self) -> bool:
        """
        Load CVE to ATT&CK technique mappings from kev_cve_attack.csv

        Returns:
            Whether loading was successful
        """
        if not os.path.exists(self.kev_cve_attack_path):
            logger.warning(
                f"KEV CVE-ATT&CK mapping CSV not found at {self.kev_cve_attack_path}"
            )
            return False

        try:
            # Load mapping CSV using pandas
            df = pd.read_csv(self.kev_cve_attack_path)
            logger.info(
                f"Successfully read {len(df)} rows from {self.kev_cve_attack_path}"
            )

            # Convert to dictionary
            self.cve_technique_map = {}

            for _, row in df.iterrows():
                # Get capability ID (CVE)
                capability_id = row.get("capability_id")
                if not capability_id or not isinstance(capability_id, str):
                    continue

                # Extract CVE ID from capability ID
                cve_matches = re.findall(r"CVE-\d+-\d+", capability_id)
                if not cve_matches:
                    continue

                cve_id = cve_matches[0]

                # Get technique ID and mapping type
                technique_id = row.get("attack_object_id")
                technique_name = row.get("attack_object_name", "")
                mapping_type = row.get("mapping_type", "")
                capability_group = row.get("capability_group", "")
                capability_description = row.get("capability_description", "")

                if not technique_id:
                    continue

                # Add to mapping
                if cve_id not in self.cve_technique_map:
                    self.cve_technique_map[cve_id] = {
                        "techniques": [],
                        "source": "kev_cve_attack_mapping",
                    }

                # Check if technique already exists
                existing_techniques = [
                    t["technique_id"]
                    for t in self.cve_technique_map[cve_id]["techniques"]
                ]

                if technique_id not in existing_techniques:
                    self.cve_technique_map[cve_id]["techniques"].append(
                        {
                            "technique_id": technique_id,
                            "name": technique_name,
                            "confidence": 0.9,  # High confidence for direct mappings
                            "mapping_type": mapping_type,
                            "capability_group": capability_group,
                            "capability_description": capability_description,
                        }
                    )

            logger.info(f"Loaded {len(self.cve_technique_map)} CVE-ATT&CK mappings")
            technique_count = sum(
                len(m["techniques"]) for m in self.cve_technique_map.values()
            )
            logger.info(f"Total technique mappings: {technique_count}")
            return True

        except Exception as e:
            logger.error(f"Error loading CVE-ATT&CK mappings: {e}")
            return False

    def get_techniques_for_cve(self, cve_id: str) -> List[Dict]:
        """
        Get techniques associated with a CVE

        Args:
            cve_id: CVE identifier

        Returns:
            List of technique dictionaries
        """
        # Normalize CVE ID format
        cve_id = cve_id.upper()
        if not cve_id.startswith("CVE-"):
            cve_id = f"CVE-{cve_id}"

        # Get direct mappings
        if cve_id in self.cve_technique_map:
            return self.cve_technique_map[cve_id]["techniques"]

        return []

    def extract_cves_from_text(self, text: str) -> List[str]:
        """
        Extract CVE IDs from text

        Args:
            text: Input text

        Returns:
            List of CVE IDs
        """
        # Simple regex pattern for CVE IDs
        cve_pattern = r"CVE-\d{4}-\d{1,7}"
        cves = re.findall(cve_pattern, text, re.IGNORECASE)

        # Normalize to uppercase
        return [cve.upper() for cve in cves]

    def get_techniques_from_text(
        self, text: str, min_confidence: float = 0.5
    ) -> List[Dict]:
        """
        Extract techniques from text based on CVE mentions

        Args:
            text: Input text
            min_confidence: Minimum confidence threshold

        Returns:
            List of technique matches
        """
        # Extract CVEs
        cves = self.extract_cves_from_text(text)

        if not cves:
            return []

        # Get techniques for each CVE
        all_techniques = []

        for cve_id in cves:
            techniques = self.get_techniques_for_cve(cve_id)

            for technique in techniques:
                if technique.get("confidence", 0) >= min_confidence:
                    # Add CVE source
                    technique["cve_id"] = cve_id
                    all_techniques.append(technique)

        return all_techniques

    def generate_cwe_technique_mappings(self, techniques: Dict) -> bool:
        """
        Placeholder to maintain API compatibility.
        Does not generate any mappings as requested.

        Args:
            techniques: Dictionary of technique data

        Returns:
            Always returns True
        """
        logger.info("CWE-technique mapping generation skipped as requested")
        return True
