# src/preprocessing/text_processor.py
import html
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF for PDF processing
import langdetect
from bs4 import BeautifulSoup
from googletrans import Translator

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"text_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)

logger = logging.getLogger("TextProcessing")


class TextProcessor:
    """Enhanced text preprocessing for better extraction results"""

    def __init__(self):
        """Initialize the text processor"""
        # Common security terms and abbreviations with their expanded forms
        self.security_terms = {
            "3des": "triple data encryption algorithm (also tdea or triple dea)",
            "aes": "advanced encryption standard",
            "des": "data encryption standard",
            "md5": "message-digest algorithm",
            "rsa": "rivest-shamir-adleman open cryptosystem",
            "sha": "secure haching algorithm",
            "csrf": "cross site request forgery",
            "dc": "differential cryptanalytics",
            "lc": "linear cryptanalytics",
            "da": "davies attack",
            "dos": "denial of service",
            "ddos": "distributed denial of service",
            "malware": "malicious software",
            "mitm": "man in the middle (also person in the middle)",
            "raas": "ransomware as a service",
            "rat": "remote access trojan",
            "rce": "remote code execution",
            "set": "social engineering toolkit",
            "sqli": "sql injection",
            "ssrf": "server side request forgery",
            "xfs": "cross frame scripting",
            "xss": "cross site scripting",
            "adr": "application detection and response",
            "amsi": "anti-malware scan interface",
            "aso": "autonomic security operations doc",
            "aspm": "application security posture management",
            "ast": "application security testing details",
            "av": "anti-virus",
            "cadr": "cloud application detection and response",
            "caasm": "cyber asset attack surface management (inventory management)",
            "casb": "cloud access security broker",
            "cdr": "cloud detection and response",
            "ciem": "cloud infrastructure entitlement management",
            "ciam": "cloud identity access management",
            "cira": "cloud investigation and response automation",
            "cnapp": "cloud native application protection platform",
            "c-scrm": "cyber supplly chain risk management link",
            "cspm": "cloud security posture management",
            "ctem": "cloud threat exposure management",
            "cwp": "cloud workload protection",
            "cwpp": "cloud workload protection platform",
            "dast": "dynamic application security testing",
            "ddr": "data detection & response",
            "dlp": "data loss prevention",
            "dspm": "data security posture management",
            "edr": "endpoint detection and response, sometimes known as endpoint threat detection and response (etdr)",
            "etdr": "see edr",
            "hids": "host based intrusion detection system (also nids for network)",
            "hips": "host intrusion prevention system",
            "iast": "interactive application security testing",
            "ids": "intrusion detection system",
            "idtr": "identity detection & response",
            "iga": "identity governance and administration",
            "ips": "intrusion protection system",
            "ispm": "identity security posture management",
            "itdr": "identity threat detection & response",
            "mdr": "managed detection and response",
            "mdft": "mobile device forensic tool",
            "mssp": "managed security services provider",
            "ndr": "network detection & response",
            "nges": "next generation endpoint security",
            "ngswg": "next generation secure web gateway",
            "nids": "network intrustion detection system",
            "nta": "network traffic analysis",
            "otspm": "operational technology security posture management link",
            "rasp": "runtime application self-protection",
            "sase": "secure access service edge",
            "sast": "static application security testing",
            "sca": "software composition analysis",
            "scap": "security content automation protocols",
            "siem": "security incident & event management",
            "soar": "security orchestration & response",
            "sse": "security services edge (a subset of sase)",
            "sspm": "saas security posture management",
            "swg": "secure web gateway link",
            "tip": "threat intelligence platform",
            "tprm": "third party risk management",
            "uba / ueba": "user and entity behavior analytics",
            "vm": "vulnerability management (also virtual machine outside of infosec)",
            "waf": "web application firewall",
            "xdr": "extended detection and response",
            "ztna": "zero trust network access",
            "apra": "australian prudential regulation authority",
            "aslr": "address space layout randomisation",
            "asvs": "(owasp) application security verification standard",
            "att&ck": "(mitre) adversarial tactics, techniques, and common knowledge",
            "bgdpl": "brazilian general data protection law (brazil)",
            "capec": "common attack pattern enumeration and classification",
            "csaf": "common security advisory framework (2.0)",
            "cis": "center for internet security link",
            "cve": "common vulnerabilities and exposures",
            "cvrf": "common vulnerability reporting framework (now csaf)",
            "cvs": "common vulnerability score",
            "cvss": "common vulnerability scoring system",
            "dss": "data security standard (see pci)",
            "epss": "exploit prediction scoring system",
            "gdpr": "general data protection regulation (europe)",
            "hipaa": "health insurance portability and accountability act",
            "iso": "international organization for standardization",
            "mitre": 'not an acronym - "a name that was meaningless and without connotations, but with an attractive feel."',
            "nvd": "national vulnerability database (usa)",
            "nist": "national institute of standards and technology (us)",
            "owasp": "open web application security project",
            "pci dss": "payment card industry data security standard",
            "pci ssc": "payment card industry security standards council",
            "pipeda": "personal information protection and electronic documents act (canada)",
            "tara": "threat agent risk assessment (methodology)",
            "samm": "software assurance maturity model (owasp) link",
            "slsa": "supply-chain levels for software artifact - link",
            "soc (1,2,3)": 'system and organization controls. see also the *"processes, teams and roles"* section',
            "2fa": "two factor authentication; see also mfa",
            "abac": "attribute based access control",
            "acl": "access control list",
            "ca": "certificate authority",
            "cors": "cross origin resource sharing",
            "doh": "dns over https",
            "dom": "document object model",
            "ftps": "ftp-ssl or ftp secure",
            "ir": "incident response",
            "jit": "just in time (saml)",
            "jwt": "json web token",
            "mfa": "multi factor authentication",
            "mtls": "mutual transport layer security",
            "oasis": "organisation for the advancement of structured information standards",
            "oauth": "open authorization",
            "otp": "one time password ( sometimes one time pad)",
            "pac": "policy as code",
            "saml": "security assertion markup language",
            "sarif": "static analysis results interchange format",
            "sftp": "ssh file transfer protocol",
            "spdx": "software package data exchange link",
            "ssh": "secure shell",
            "ssl": "secure sockets layer",
            "sso": "single sign-on",
            "tlp": "traffic light protocol",
            "tls": "transport layer security",
            "u2f": "universal two factor",
            "wep": "wired equivalent privacy (protocol)",
            "wpa": "wi-fi protected access (protocol)",
            "wps": "wi-fi protected setup (standard)",
            "a&a": "assessment and authorization",
            "ccsp": "certified cloud security professional (isc2)",
            "cdc": "cyber defense center",
            "cert": "computer emergency response team",
            "ciso": "chief information security officer",
            "cissp": "certified information systems security professional",
            "coc": "cybersecurity operations center",
            "cpp": "certified protection professional",
            "cso": "chief security officer (role)",
            "eces": "certified encryption specialist",
            "first": "forum of incident response and security teams",
            "niccs": "national initiative for cybersecurity careers and studies",
            "nice": "niccs workforce framework for cybersecurity",
            "oscp": "offensive security certified professional",
            "soc": "security operations center",
            "secops": "organizational term. collaboration between security and operations teams by sharing security responsibilities",
            "apt": "advanced persistent threat",
            "authn": "authentication",
            "authz": "authorization",
            "bas": "breach & attack simulation",
            "bcp": "business continuity plan",
            "bec": "business email compromise",
            "bgh": "big game hunting",
            "bia": "business impact analysis",
            "bsimm": "building security in maturity model",
            "c2": "command & control",
            "captcha": "completely automated public turing test to tell computers and humans apart",
            "cia": "confidentiality; integrity; availability",
            "cisa": "cybersecurity and infrastructure security agency | certified information systems auditor",
            "coa": "course of action",
            "cta": "cyber threat intelligence",
            "iam": "identity & access management",
            "ioa": "indicators of attack",
            "ioc": "indicators of compromise",
            "malops": "malicious operations",
            "mttr": "mean time to resolve",
            "pam": "privileged access management",
            "rbac": "role based access control",
            "sdlc": "software development lifecycle (also sometimes system development lifecycle)",
            "sd-wan": "software defined wide area network",
            "sku": "stock keeping unit (unique identificaiton that definees an element)",
            "sra": "security response automation",
            "sss": "stack smashing protector (compilers)",
            "swot": "strengths, weaknesses, opportunities, and threats (swot analysis)",
            "ti": "threat intelligence",
            "ttp": "tactics, techniques, and procedures",
            "uac": "user access control",
            "vap": "very attacked person",
            "vpn": "virtual private network",
            "yara": "yet another ridiculous acronym - rule-based tool for malware analysis link",
            "yara-l": "yara for logs (chronicle aka secops)",
            "ccm": "cloud controls matrix",
            "nhi": "non human identity",
            "nms": "network management system",
            "nrt": "near real time",
            "tpp": "third party payment provider",
            "capp": "controlled access protection profile",
            "cissp": "certified information systems security professional (isc2)",
            "cmf": "collection management framework",
            "csa": "(1) cloud security alliance (2) continuous security assessment",
            "csp": "content security policy",
            "ctf": "capture the flag",
            "cti": "cyber threat intelligence",
            "cwe": "common weakness enumeration",
            "dep": "data execution prevention",
            "dfir": "digital forensics and incident response",
            "dkim": "domainkeys identified mail",
            "dls": "dedicated leak site",
            "dmarc": "domain-based message authentication, reporting & conformance",
            "dnssec": "domain name system security extensions",
            "dread": "damage; reproducability; exploitability; affected users; discoverability",
            "easm": "externam attack surface management",
            "eicar": "european institute for computer antivirus research",
            "epp": "endpoint protection platform",
            "fair": "factor analysis of information risk",
            "fam": "file access monitoring",
            "fido": "fast identity online",
            "fim": "file integrity monitoring",
            "fpc": "full packet capture",
            "gcm": "galois/counter mode",
            "gpg": "gnupg",
            "grc": "governance, risk & compliance",
            "hsm": "hardware security module",
            "hsts": "http strict transfer protocol",
            "idam": "identity & access management",
            "idor": "insecure direct object reference",
            "idp": "identity provider",
            "ietf": "internet engineering task force",
            "ipe": "intelligence preperation of the environment",
            "ipsec": "internet protocol security",
            "irm": "integrated risk management",
            "irp": "incident response playbook",
            "isc2": "international information system security certification consortium",
            "isms": "information security management system",
            "iss": "information system security",
            "kcm": "kill chain model",
            "langsec": "language security",
            "lfi": "local file inclusion",
            "lolbin": "living off the land binary (also lolscripts, lolbas)",
            "nac": "network access control / also nacl (network access control list)",
            "ndb": "notifiable data breache(s)",
            "ngci": "next generation cyber infrastructure",
            "ngfw": "next generation firewall",
            "odoh": "oblivious dns over https",
            "oidc": "openid connect",
            "opsec": "operational security",
            "oscal": "open security controls assessment language",
            "osint": "open source intelligence",
            "pasta": "process for attack simulation & threat analysis",
            "pcd": "payment card data",
            "pgp": "pretty good privacy. see also gpg",
            "pfs": "perfect forward secrecy",
            "ptes": "penetration testing execution standard",
            "pup": "potentially unwanted program",
            "rfc": "request for comments",
            "rop": "return-oriented programming",
            "rp": "return pointer",
            "rtr": "rapid threat response",
            "sabsa": "sherwood applied business security architecture",
            "sans": "sysadmin, audit, network, and security",
            "saq": "self-assessment questionnaire",
            "scim": "system for cross-domain identity management",
            "ssdlc": "secure software development lifecycle",
            "seccomp": "secure computing",
            "sfp": "saved frame pointer",
            "soa": "statemenet of applicability",
            "sox": "sarbanes-oxley act",
            "spf": "sender policy framework",
            "sri": "sub-resource integrity",
            "ssvc": "stakeholder-specific vulnerability categorization",
            "stig": "security technical implementation guide",
            "stix": "structured threat information expression",
            "stride": "spoofing; tampering; repudiation; information disclosure; denial of service; elevation of privilege",
            "taxii": "trusted automated exchange of intelligence information",
            "togaf": "the open group architecture framework",
            "xacml": "extensible access control markup language",
            "xxe": "xml external entity",
        }

        self.translator = Translator()

    def preprocess(
        self, text: str, content_type: str = "text", file_path: str = None
    ) -> str:
        """Preprocess text based on content type

        Args:
            text: Input text
            content_type: Content type (text, html, markdown)

        Returns:
            Preprocessed text
        """
        # Handle different content types
        if content_type == "pdf" and file_path:
            text = self._extract_pdf_text(file_path)
        elif content_type == "html":
            text = self._clean_html(text)
        elif content_type == "markdown":
            text = self._clean_markdown(text)

        # Detect language and translate if not English
        lang = self._detect_language(text)
        if lang != "en":
            text = self._translate_to_english(text, lang)

        # Common preprocessing steps
        text = self._normalize_whitespace(text)
        text = self._expand_abbreviations(text)
        text = self._normalize_cve_references(text)

        # Segment text into meaningful units
        segments = self._segment_text(text)

        # Rejoin segments with proper spacing
        return "\n\n".join(segments)

    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return ""

    def _detect_language(self, text: str) -> str:
        """Detect language of text"""
        try:
            return langdetect.detect(text)
        except:
            return "en"

    def _translate_to_english(self, text: str, source_lang: str) -> str:
        """Translate text to English"""
        try:
            if len(text) > 5000:
                # Break into chunks for large texts
                chunks = [text[i : i + 5000] for i in range(0, len(text), 5000)]
                translated_chunks = [
                    self.translator.translate(chunk, src=source_lang, dest="en").text
                    for chunk in chunks
                ]
                return " ".join(translated_chunks)
            else:
                return self.translator.translate(text, src=source_lang, dest="en").text
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text  # Return original text if translation fails

    def _segment_text(self, text: str) -> List[str]:
        """Segment text into meaningful units"""
        # Split by paragraphs first
        paragraphs = [p for p in text.split("\n\n") if p.strip()]

        segments = []
        for para in paragraphs:
            # For very long paragraphs, try to break at sentence boundaries
            if len(para) > 1000:
                # Split on sentence boundaries
                sentences = re.split(r"(?<=[.!?])\s+", para)
                current_segment = ""

                for sentence in sentences:
                    if len(current_segment) + len(sentence) < 1000:
                        current_segment += sentence + " "
                    else:
                        segments.append(current_segment.strip())
                        current_segment = sentence + " "

                if current_segment:
                    segments.append(current_segment.strip())
            else:
                segments.append(para)

        return segments

    def _clean_html(self, html_text: str) -> str:
        """Remove HTML tags and extract meaningful text"""
        try:
            # Use BeautifulSoup for better HTML handling
            soup = BeautifulSoup(html_text, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()

            # Get text
            text = soup.get_text()

            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = "\n".join(chunk for chunk in chunks if chunk)

            return text
        except Exception as e:
            # Fallback to simple regex if BeautifulSoup fails
            return re.sub(r"<[^>]+>", " ", html_text)

    def _clean_markdown(self, markdown_text: str) -> str:
        """Clean markdown formatting"""
        # Remove code blocks
        text = re.sub(r"```.*?```", " ", markdown_text, flags=re.DOTALL)

        # Remove inline code
        text = re.sub(r"`.*?`", " ", text)

        # Remove headers
        text = re.sub(r"#{1,6}\s+", " ", text)

        # Remove markdown links but keep the text
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

        # Remove images
        text = re.sub(r"!\[.*?\]\(.*?\)", " ", text)

        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text"""
        # Replace multiple spaces with a single space
        text = re.sub(r"\s+", " ", text)

        # Ensure proper spacing after punctuation
        text = re.sub(r"([.,;!?])(\w)", r"\1 \2", text)

        return text.strip()

    def _expand_abbreviations(self, text: str) -> str:
        """Expand common security abbreviations"""
        # Convert text to lowercase for matching
        text_lower = text.lower()

        # Find and replace abbreviations
        for abbr, expansion in self.security_terms.items():
            # Replace whole-word matches only
            pattern = r"\b" + re.escape(abbr) + r"\b"
            text_lower = re.sub(pattern, expansion, text_lower)

        return text_lower

    def _normalize_cve_references(self, text: str) -> str:
        """Normalize CVE references to standard format"""
        # Find potential CVE references
        cve_pattern = r"\b(?:cve[-\s]?(\d{4})[-\s]?(\d{1,7}))\b"

        def replace_cve(match):
            year = match.group(1)
            id_num = match.group(2)
            return f"CVE-{year}-{id_num}"

        # Replace with normalized format
        text = re.sub(cve_pattern, replace_cve, text, flags=re.IGNORECASE)

        return text
