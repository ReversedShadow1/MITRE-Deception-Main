"""
ATT&CK Data Manager
------------------
Handles loading and processing MITRE ATT&CK data, keywords, and KEV integration.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
import requests

# Configure logging
logger = logging.getLogger("ATTCKDataManager")


class ATTCKDataLoader:
    """
    Handles loading and managing MITRE ATT&CK data, keywords, and KEV-related data
    """

    def __init__(self, data_dir: str = "data", auto_load: bool = True):
        """
        Initialize the data loader

        Args:
            data_dir: Directory for data storage
            auto_load: Whether to automatically load data on initialization
        """
        self.data_dir = data_dir

        # Data storage
        self.techniques = {}  # ATT&CK techniques
        self.technique_keywords = {}  # Technique keywords
        self.cve_technique_map = {}  # CVE to technique mappings
        self.kev_data = {}  # KEV catalog data
        self.cwe_technique_map = {}  # CWE to technique mappings

        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

        # File paths
        self.enterprise_attack_path = os.path.join(data_dir, "enterprise-attack.json")
        self.keywords_path = os.path.join(data_dir, "technique_keywords.json")
        self.cve_mapping_path = os.path.join(data_dir, "cve_attack_mappings.csv")
        self.kev_csv_path = os.path.join(data_dir, "kev_catalog.csv")
        self.cve_technique_json_path = os.path.join(data_dir, "cve_technique_map.json")

        # Status flags
        self.attack_loaded = False
        self.keywords_loaded = False
        self.mappings_loaded = False
        self.kev_loaded = False

        # Auto-load data if requested
        if auto_load:
            self.load_data()

    def load_data(self, force_download: bool = False) -> bool:
        """
        Load all required data

        Args:
            force_download: Whether to force download of data even if files exist

        Returns:
            Success status
        """
        # Load Enterprise ATT&CK data
        attack_loaded = self.load_enterprise_attack(force_download)

        # Load technique keywords
        keywords_loaded = False
        if attack_loaded:
            keywords_loaded = self.load_technique_keywords()

            # Generate keywords if not loaded and techniques are available
            if not keywords_loaded and self.techniques:
                logger.info("Generating basic technique keywords...")
                self._generate_basic_keywords()
                keywords_loaded = True

        # Load CVE-ATT&CK mappings and KEV data
        self._check_csv_files()  # Check and convert Excel files if needed
        mappings_loaded = self.load_cve_attack_mappings()
        kev_loaded = self.load_kev_data()

        # Store status
        self.attack_loaded = attack_loaded
        self.keywords_loaded = keywords_loaded
        self.mappings_loaded = mappings_loaded
        self.kev_loaded = kev_loaded

        # Generate CWE-technique mappings if both KEV and ATT&CK data are available
        # Using actual data presence rather than flags
        if self.kev_data and self.techniques:
            self.generate_cwe_technique_mappings()

        return attack_loaded and keywords_loaded

    def _check_csv_files(self) -> None:
        """Check for CSV files and convert from Excel if needed"""
        # Check for CVE-ATT&CK mapping file
        if not os.path.exists(self.cve_mapping_path):
            # Try to create it from Excel data if available
            excel_path = os.path.join(self.data_dir, "cve_attack_mappings.xlsx")
            if os.path.exists(excel_path):
                try:
                    logger.info(f"Converting Excel mapping file to CSV: {excel_path}")
                    df = pd.read_excel(excel_path)
                    df.to_csv(self.cve_mapping_path, index=False)
                    logger.info(f"Converted Excel to CSV: {self.cve_mapping_path}")
                except Exception as e:
                    logger.error(f"Failed to convert Excel mapping file: {str(e)}")

        # Check for KEV catalog file
        if not os.path.exists(self.kev_csv_path):
            # Try to create it from Excel if available
            kev_excel_path = os.path.join(self.data_dir, "kev_catalog.xlsx")
            if os.path.exists(kev_excel_path):
                try:
                    logger.info(f"Converting KEV Excel file to CSV: {kev_excel_path}")
                    df = pd.read_excel(kev_excel_path)
                    df.to_csv(self.kev_csv_path, index=False)
                    logger.info(f"Converted KEV Excel to CSV: {self.kev_csv_path}")
                except Exception as e:
                    logger.error(f"Failed to convert KEV Excel file: {str(e)}")

    def load_enterprise_attack(self, force_download: bool = False) -> bool:
        """
        Load Enterprise ATT&CK data from local file or download if needed

        Args:
            force_download: Whether to force download even if file exists

        Returns:
            Success status
        """
        # Check if file exists and download if needed
        if not os.path.exists(self.enterprise_attack_path) or force_download:
            logger.info("Downloading Enterprise ATT&CK data...")

            try:
                # Download latest data from MITRE GitHub repo
                url = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
                response = requests.get(url)
                response.raise_for_status()

                # Save to file
                with open(self.enterprise_attack_path, "w", encoding="utf-8") as f:
                    f.write(response.text)

                logger.info(
                    f"Downloaded Enterprise ATT&CK data to {self.enterprise_attack_path}"
                )
            except Exception as e:
                logger.error(f"Failed to download Enterprise ATT&CK data: {str(e)}")
                return False

        # Load data from file
        try:
            with open(self.enterprise_attack_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Parse the data
            self._parse_attack_data(data)
            return True
        except Exception as e:
            logger.error(f"Failed to load Enterprise ATT&CK data: {str(e)}")
            return False

    def _parse_attack_data(self, data: Dict) -> None:
        """
        Parse ATT&CK STIX data to extract technique information

        Args:
            data: ATT&CK STIX data
        """
        # Temporary storage for tactics
        tactics = {}
        techniques = {}

        # Ensure data has the expected structure
        if "objects" not in data:
            logger.error("Invalid ATT&CK data format")
            return

        # First pass - collect tactics
        for obj in data["objects"]:
            if obj.get("type") == "x-mitre-tactic":
                tactic_id = None

                # Extract tactic ID
                for ref in obj.get("external_references", []):
                    if ref.get("source_name") == "mitre-attack":
                        tactic_id = ref.get("external_id")
                        break

                if tactic_id:
                    tactics[obj["id"]] = {
                        "id": tactic_id,
                        "name": obj.get("name", ""),
                        "shortname": obj.get("x_mitre_shortname", ""),
                        "url": f"https://attack.mitre.org/tactics/{tactic_id}/",
                    }

        logger.info(f"Parsed {len(tactics)} ATT&CK tactics")

        # Second pass - collect techniques
        for obj in data["objects"]:
            if obj.get("type") == "attack-pattern":
                # Extract technique ID
                tech_id = None
                tech_url = None

                for ref in obj.get("external_references", []):
                    if ref.get("source_name") == "mitre-attack":
                        tech_id = ref.get("external_id")
                        tech_url = ref.get("url")
                        break

                if not tech_id:
                    continue

                # Get related tactics
                tactic_refs = []
                tactic_names = []
                tactic_shortnames = []

                for phase in obj.get("kill_chain_phases", []):
                    if phase.get("kill_chain_name") == "mitre-attack":
                        phase_name = phase.get("phase_name")

                        # Find matching tactic
                        for tactic_id, tactic in tactics.items():
                            if tactic["shortname"] == phase_name:
                                tactic_refs.append(tactic_id)
                                tactic_names.append(tactic["name"])
                                tactic_shortnames.append(phase_name)

                # Create technique entry
                techniques[tech_id] = {
                    "id": tech_id,
                    "name": obj.get("name", ""),
                    "description": obj.get("description", ""),
                    "tactics": tactic_names,
                    "tactic_refs": tactic_refs,
                    "tactic_shortnames": tactic_shortnames,
                    "platforms": obj.get("x_mitre_platforms", []),
                    "permissions_required": obj.get("x_mitre_permissions_required", []),
                    "data_sources": obj.get("x_mitre_data_sources", []),
                    "defense_bypassed": obj.get("x_mitre_defense_bypassed", []),
                    "detection": obj.get("x_mitre_detection", ""),
                    "url": tech_url
                    or f"https://attack.mitre.org/techniques/{tech_id}/",
                }

        self.techniques = techniques
        logger.info(f"Parsed {len(techniques)} ATT&CK techniques")

    def load_technique_keywords(self) -> bool:
        """
        Load technique keywords from file

        Returns:
            Success status
        """
        if os.path.exists(self.keywords_path):
            try:
                with open(self.keywords_path, "r", encoding="utf-8") as f:
                    self.technique_keywords = json.load(f)
                logger.info(
                    f"Loaded {len(self.technique_keywords)} technique keyword mappings"
                )
                return True
            except Exception as e:
                logger.error(f"Failed to load technique keywords: {str(e)}")
                return False
        else:
            logger.warning(f"Technique keywords file not found at {self.keywords_path}")
            return False

    def _generate_basic_keywords(self) -> None:
        """
        Generate basic keywords for techniques from their names and descriptions
        """
        if not self.techniques:
            logger.error("Cannot generate keywords - no techniques loaded")
            return

        # Stop words to filter out
        stop_words = {
            "the",
            "and",
            "for",
            "with",
            "this",
            "that",
            "from",
            "their",
            "they",
            "have",
            "has",
            "been",
            "were",
            "when",
            "where",
            "will",
            "what",
            "who",
            "how",
            "some",
            "such",
            "than",
            "then",
            "these",
            "those",
            "there",
            "about",
        }

        keywords = {}

        for tech_id, tech in self.techniques.items():
            # Get technique name and description
            name = tech.get("name", "")
            desc = tech.get("description", "")

            # Extract first sentence of description (usually the most relevant)
            first_sentence = desc.split(".")[0] if desc else ""

            # Extract keywords from name (lower case, split on spaces and punctuation)
            name_words = re.findall(r"\b[a-zA-Z0-9]+\b", name.lower())
            name_words = [w for w in name_words if len(w) > 3 and w not in stop_words]

            # Extract keywords from first sentence
            desc_words = re.findall(r"\b[a-zA-Z0-9]+\b", first_sentence.lower())
            desc_words = [w for w in desc_words if len(w) > 3 and w not in stop_words]

            # Include full technique name
            tech_keywords = [name.lower()]

            # Include multi-word phrases from name
            name_phrases = []
            name_parts = re.split(r"[/\(\)\[\]{}:]", name)
            for part in name_parts:
                part = part.strip()
                if part and " " in part:
                    name_phrases.append(part.lower())

            # Add relevant words from name and description
            tech_keywords.extend(name_phrases)
            tech_keywords.extend(name_words)
            tech_keywords.extend(desc_words)

            # Remove duplicates and keep only unique keywords
            tech_keywords = list(set(tech_keywords))

            # Store keywords for this technique
            keywords[tech_id] = tech_keywords

        self.technique_keywords = keywords

        # Save to file
        try:
            with open(self.keywords_path, "w", encoding="utf-8") as f:
                json.dump(keywords, f, indent=2)
            logger.info(f"Generated and saved keywords for {len(keywords)} techniques")
        except Exception as e:
            logger.error(f"Failed to save generated keywords: {str(e)}")

    def load_kev_data(self) -> bool:
        """
        Load KEV catalog data from CSV

        Returns:
            Whether loading was successful
        """
        if not os.path.exists(self.kev_csv_path):
            logger.warning(f"KEV catalog CSV not found at {self.kev_csv_path}")
            return False

        try:
            # Load KEV CSV using pandas for better column handling
            df = pd.read_csv(self.kev_csv_path)

            # Convert to dictionary
            self.kev_data = {}

            for _, row in df.iterrows():
                # Get CVE ID - handle different column names
                cve_id = None
                if "cveID" in row:
                    cve_id = row.get("cveID")
                elif "cveId" in row:
                    cve_id = row.get("cveId")
                else:
                    # Try to find column containing CVE
                    for col in row.index:
                        if isinstance(row[col], str) and row[col].startswith("CVE-"):
                            cve_id = row[col]
                            break

                if not cve_id:
                    continue

                # Ensure CVE-ID format
                if not cve_id.startswith("CVE-"):
                    cve_id = f"CVE-{cve_id}"

                # Extract CWEs if available
                cwes = []
                for cwe_col in ["cwes", "cwe", "CWE"]:
                    if cwe_col in row and pd.notna(row[cwe_col]):
                        # Handle potential list format or string format
                        cwe_value = row[cwe_col]
                        if isinstance(cwe_value, str):
                            # Split by commas if multiple CWEs
                            if "," in cwe_value:
                                cwes = [cwe.strip() for cwe in cwe_value.split(",")]
                            else:
                                cwes = [cwe_value.strip()]
                        break

                # Create entry - handling different column names
                entry = {"cwes": cwes}

                # Map common KEV columns with fallbacks
                column_mappings = {
                    "vendorProject": [
                        "vendorProject",
                        "vendor",
                        "vendorproject",
                        "vendor_project",
                    ],
                    "product": ["product", "products", "affected_product"],
                    "vulnerabilityName": [
                        "vulnerabilityName",
                        "vulnerability_name",
                        "vuln_name",
                        "name",
                    ],
                    "dateAdded": ["dateAdded", "date_added", "date", "added_date"],
                    "shortDescription": [
                        "shortDescription",
                        "description",
                        "short_description",
                        "desc",
                    ],
                    "requiredAction": [
                        "requiredAction",
                        "required_action",
                        "action",
                        "mitigation",
                    ],
                    "dueDate": ["dueDate", "due_date", "due", "remediation_deadline"],
                    "knownRansomwareCampaignUse": [
                        "knownRansomwareCampaignUse",
                        "ransomware",
                        "ransomware_use",
                    ],
                }

                # Find and add values using column mappings
                for key, possible_columns in column_mappings.items():
                    for col in possible_columns:
                        if col in row and pd.notna(row[col]):
                            entry[key] = row[col]
                            break

                    # Set default if not found
                    if key not in entry:
                        entry[key] = ""

                # Store entry
                self.kev_data[cve_id] = entry

            logger.info(f"Loaded {len(self.kev_data)} entries from KEV catalog")
            return True

        except Exception as e:
            logger.error(f"Error loading KEV data: {str(e)}")
            return False

    def load_cve_attack_mappings(self) -> bool:
        """
        Load CVE to ATT&CK technique mappings from CSV

        Returns:
            Whether loading was successful
        """
        # First try to load from JSON if available (faster)
        if os.path.exists(self.cve_technique_json_path):
            try:
                with open(self.cve_technique_json_path, "r", encoding="utf-8") as f:
                    self.cve_technique_map = json.load(f)

                logger.info(
                    f"Loaded {len(self.cve_technique_map)} CVE-technique mappings from JSON"
                )
                return True
            except Exception as e:
                logger.error(
                    f"Error loading CVE-technique mappings from JSON: {str(e)}"
                )
                # Continue to CSV loading as fallback

        if not os.path.exists(self.cve_mapping_path):
            logger.warning(
                f"CVE-ATT&CK mapping CSV not found at {self.cve_mapping_path}"
            )
            return False

        try:
            # Load mapping CSV using pandas
            df = pd.read_csv(self.cve_mapping_path)

            # Convert to dictionary
            self.cve_technique_map = {}

            for _, row in df.iterrows():
                # Determine column names based on what's available
                capability_id_col = None
                technique_id_col = None
                technique_name_col = None
                mapping_type_col = None

                # Find capability_id column (contains CVE)
                for col in df.columns:
                    if "capability" in col.lower() and "id" in col.lower():
                        capability_id_col = col
                        break

                # Find technique_id column
                for col in df.columns:
                    if (
                        "attack" in col.lower()
                        and "id" in col.lower()
                        or "object_id" in col.lower()
                    ):
                        technique_id_col = col
                        break

                # Find technique_name column
                for col in df.columns:
                    if (
                        "attack" in col.lower()
                        and "name" in col.lower()
                        or "object_name" in col.lower()
                    ):
                        technique_name_col = col
                        break

                # Find mapping_type column
                for col in df.columns:
                    if "mapping" in col.lower() and "type" in col.lower():
                        mapping_type_col = col
                        break

                # Skip if required columns not found
                if not capability_id_col or not technique_id_col:
                    logger.warning("Required columns not found in CVE mapping CSV")
                    continue

                # Get CVE ID (from capability_id column)
                capability_id = row.get(capability_id_col, "")
                if not capability_id or not isinstance(capability_id, str):
                    continue

                # Extract CVE ID from the capability ID field
                cve_parts = [
                    part for part in capability_id.split() if part.startswith("CVE-")
                ]
                if not cve_parts:
                    continue

                cve_id = cve_parts[0]

                # Get technique ID and name
                technique_id = row.get(technique_id_col, "")
                technique_name = (
                    row.get(technique_name_col, "") if technique_name_col else ""
                )
                mapping_type = row.get(mapping_type_col, "") if mapping_type_col else ""

                if not technique_id:
                    continue

                # Add to mapping
                if cve_id not in self.cve_technique_map:
                    self.cve_technique_map[cve_id] = {
                        "techniques": [],
                        "source": "cve_attack_mapping",
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
                        }
                    )

            logger.info(f"Loaded {len(self.cve_technique_map)} CVE-ATT&CK mappings")

            # Save to JSON for easier loading next time
            try:
                with open(self.cve_technique_json_path, "w", encoding="utf-8") as f:
                    json.dump(self.cve_technique_map, f, indent=2)

                logger.info(
                    f"Saved CVE-technique map to {self.cve_technique_json_path}"
                )
            except Exception as e:
                logger.error(f"Failed to save CVE-technique map: {str(e)}")

            return True

        except Exception as e:
            logger.error(f"Error loading CVE-ATT&CK mappings: {str(e)}")
            return False

    def generate_cwe_technique_mappings(self) -> bool:
        """
        Generate mappings from CWEs to ATT&CK techniques based on common patterns

        Returns:
            Whether generation was successful
        """
        # Check directly if needed data is available instead of relying on status flags
        if not self.kev_data or not self.techniques:
            logger.warning(
                "KEV or ATT&CK data not loaded, cannot generate CWE-technique mappings"
            )
            return False

        try:
            # CWE to potential technique mappings (based on common patterns)
            cwe_mappings = {
                # Memory-related CWEs
                "CWE-119": ["T1055", "T1185"],  # Buffer overflow
                "CWE-120": ["T1055", "T1185"],  # Buffer overflow
                "CWE-122": ["T1055", "T1185"],  # Heap buffer overflow
                "CWE-125": ["T1055", "T1186"],  # Out-of-bounds read
                "CWE-787": ["T1055", "T1565"],  # Out-of-bounds write
                # Code execution CWEs
                "CWE-78": ["T1059"],  # OS Command Injection
                "CWE-94": ["T1059"],  # Code Injection
                # File-related CWEs
                "CWE-22": ["T1083", "T1082"],  # Path Traversal
                "CWE-23": ["T1083", "T1082"],  # Path Traversal
                "CWE-36": ["T1083", "T1082"],  # Absolute Path Traversal
                "CWE-73": ["T1083", "T1082"],  # External Control of File Name
                # Authentication CWEs
                "CWE-287": ["T1110", "T1212"],  # Improper Authentication
                "CWE-294": ["T1110", "T1212"],  # Authentication Bypass
                "CWE-295": ["T1553", "T1555"],  # Certificate Validation
                "CWE-306": ["T1212", "T1550"],  # Missing Authentication
                # Privilege escalation CWEs
                "CWE-269": ["T1068", "T1548"],  # Privilege Management
                "CWE-274": ["T1068", "T1548"],  # Privilege Context
                "CWE-264": ["T1068", "T1548"],  # Permissions and Privileges
                # Web-related CWEs
                "CWE-79": ["T1189", "T1203"],  # XSS
                "CWE-89": ["T1190", "T1203"],  # SQL Injection
                # Information disclosure CWEs
                "CWE-200": ["T1552", "T1005"],  # Information Exposure
                "CWE-203": ["T1552", "T1005"],  # Information Disclosure
                "CWE-209": ["T1552", "T1005"],  # Error Information Leak
                # Deserialization CWEs
                "CWE-502": ["T1059", "T1203"],  # Deserialization
                # Configuration CWEs
                "CWE-16": ["T1552", "T1555"],  # Configuration
                "CWE-276": ["T1222"],  # Incorrect Permission Assignment
                "CWE-284": ["T1222", "T1222"],  # Access Control
                # Default/hardcoded credentials CWEs
                "CWE-798": ["T1552", "T1555"],  # Hardcoded Credentials
                "CWE-259": ["T1552", "T1555"],  # Hardcoded Password
                # SSRF CWEs
                "CWE-918": ["T1190"],  # SSRF
            }

            # Extend with more fine-grained mappings based on technique descriptions
            for tech_id, tech_data in self.techniques.items():
                tech_desc = tech_data.get("description", "").lower()

                # Add mappings based on technique descriptions
                if "buffer" in tech_desc and "overflow" in tech_desc:
                    self._add_cwe_technique_mapping(
                        ["CWE-119", "CWE-120", "CWE-122"], tech_id
                    )

                if "command injection" in tech_desc or "command execution" in tech_desc:
                    self._add_cwe_technique_mapping(["CWE-78", "CWE-77"], tech_id)

                if "sql injection" in tech_desc:
                    self._add_cwe_technique_mapping(["CWE-89"], tech_id)

                if "cross-site" in tech_desc or "xss" in tech_desc:
                    self._add_cwe_technique_mapping(["CWE-79"], tech_id)

                if "path traversal" in tech_desc:
                    self._add_cwe_technique_mapping(
                        ["CWE-22", "CWE-23", "CWE-36", "CWE-73"], tech_id
                    )

                if (
                    "privilege escalation" in tech_desc
                    or "privilege elevation" in tech_desc
                ):
                    self._add_cwe_technique_mapping(
                        ["CWE-269", "CWE-274", "CWE-264"], tech_id
                    )

                if "deserialization" in tech_desc:
                    self._add_cwe_technique_mapping(["CWE-502"], tech_id)

            # Add basic mappings
            for cwe, techniques in cwe_mappings.items():
                for tech_id in techniques:
                    self._add_cwe_technique_mapping([cwe], tech_id)

            logger.info(
                f"Generated {sum(len(techs) for techs in self.cwe_technique_map.values())} CWE-technique mappings"
            )
            return True

        except Exception as e:
            logger.error(f"Error generating CWE-technique mappings: {str(e)}")
            return False

    def _add_cwe_technique_mapping(self, cwes: List[str], technique_id: str) -> None:
        """
        Add mapping from CWE to technique

        Args:
            cwes: List of CWE IDs
            technique_id: ATT&CK technique ID
        """
        for cwe in cwes:
            # Normalize CWE format
            if not cwe.startswith("CWE-"):
                cwe = f"CWE-{cwe}"

            if cwe not in self.cwe_technique_map:
                self.cwe_technique_map[cwe] = []

            if technique_id not in self.cwe_technique_map[cwe]:
                self.cwe_technique_map[cwe].append(technique_id)

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
        direct_techniques = []
        if cve_id in self.cve_technique_map:
            direct_techniques = self.cve_technique_map[cve_id]["techniques"]

        # Get indirect mappings from CWEs if available
        indirect_techniques = []
        if self.kev_loaded and cve_id in self.kev_data:
            cwes = self.kev_data[cve_id].get("cwes", [])

            for cwe in cwes:
                # Normalize CWE format
                if not cwe.startswith("CWE-"):
                    cwe = f"CWE-{cwe}"

                # Get techniques for this CWE
                if cwe in self.cwe_technique_map:
                    for tech_id in self.cwe_technique_map[cwe]:
                        # Check if already in direct techniques
                        existing_ids = [t["technique_id"] for t in direct_techniques]

                        if tech_id not in existing_ids:
                            indirect_techniques.append(
                                {
                                    "technique_id": tech_id,
                                    "confidence": 0.6,  # Lower confidence for CWE-derived mappings
                                    "source": f"cwe_derived:{cwe}",
                                }
                            )

        # Combine direct and indirect techniques
        return direct_techniques + indirect_techniques

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

    def update_keywords(self, tech_id: str, keywords: List[str]) -> bool:
        """
        Update keywords for a specific technique

        Args:
            tech_id: Technique ID
            keywords: New keywords

        Returns:
            Success status
        """
        if tech_id not in self.techniques:
            logger.error(f"Technique {tech_id} not found")
            return False

        # Update keywords
        self.technique_keywords[tech_id] = keywords

        # Save to file
        try:
            with open(self.keywords_path, "w", encoding="utf-8") as f:
                json.dump(self.technique_keywords, f, indent=2)
            logger.info(f"Updated keywords for technique {tech_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save updated keywords: {str(e)}")
            return False

    def get_technique_by_id(self, tech_id: str) -> Dict:
        """
        Get technique data by ID

        Args:
            tech_id: Technique ID

        Returns:
            Technique data dictionary or empty dict if not found
        """
        return self.techniques.get(tech_id, {})

    def get_techniques_for_tactic(self, tactic: str) -> List[Dict]:
        """
        Get all techniques for a specific tactic

        Args:
            tactic: Tactic name or shortname

        Returns:
            List of technique dictionaries
        """
        tactic = tactic.lower()
        results = []

        for tech_id, tech in self.techniques.items():
            # Check if tactic matches any tactic name or shortname
            tactic_names = [t.lower() for t in tech.get("tactics", [])]
            tactic_shortnames = [t.lower() for t in tech.get("tactic_shortnames", [])]

            if tactic in tactic_names or tactic in tactic_shortnames:
                results.append(
                    {
                        "id": tech_id,
                        "name": tech.get("name", ""),
                        "description": tech.get("description", ""),
                        "url": tech.get("url", ""),
                    }
                )

        return results
