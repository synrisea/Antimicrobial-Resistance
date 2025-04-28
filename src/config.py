"""
Configuration settings and constants for the AMR data processing project.
"""

# --- List of Antibiotics and Aliases ---
# Comprehensive list including variations
ANTIBIOTICS = [
    "3GCREC", "3GCRKP", "fluoroquinolones", "aminoglycosides", "aminopenicillins",
    "carbapenems", "vancomycin", "piperacillin (+ tazobactam)", "ceftazidime",
    "meticillin", "macrolides", "penicillin", "third-generation cephalosporins",
    "penicillins", "piperacillin(+tazobactam)", "piperacillin+tazobactam",
    "piperacillin±tazobactam", "gentamicin", "piperacillin (+ tazobactam)",
    "piperacillin", "piperacillin + tazobactam", "aminoglycoside"
]

# Dictionary for normalizing antibiotic names
ABX_ALIASES = {
    "piperacillin(+tazobactam)": "piperacillin+tazobactam",
    "piperacillin±tazobactam": "piperacillin+tazobactam",
    "piperacillin (+ tazobactam)": "piperacillin+tazobactam",
    "piperacillin" : "piperacillin+tazobactam", # Normalize standalone piperacillin to the combination
    "penicillin": "penicillins",
    "piperacillin + tazobactam" : "piperacillin+tazobactam",
    "aminoglycoside" : "aminoglycosides"
}

# --- List of Countries ---
# List of expected countries
COUNTRIES_LIST = [
    'Belgium', 'Finland', 'Sweden', 'Ireland', 'Denmark', 'Netherlands',
    'United Kingdom', 'Germany', 'Norway', 'Malta', 'France', 'Austria',
    'Czech Republic', 'Slovakia', 'Portugal', 'Slovenia', 'Spain', 'Hungary',
    'Latvia', 'Bulgaria', 'Italy', 'Romania', 'Cyprus', 'Poland', 'Croatia',
    'Lithuania', 'Greece', 'Estonia', 'Iceland', 'Luxembourg'
]
# Normalized set for efficient lookup (lowercase)
NORMALIZED_COUNTRIES = {c.lower() for c in COUNTRIES_LIST}

# --- Patterns for Column Identification ---
# Potential names for columns containing the number of isolates (N)
N_PATTERNS = [
    'Number of.1', 'Number of isolates', 'Number of \ntested isolates',
    'Number of tested isolates',
    'Number of \n3GCREC included in analysis/ total number of 3GCREC',
    'Number of\n3GCRKP\nincluded in\nanalysis/total\nnumber of\n3GCRKP',
    'N', 'N.1', 'N.2', 'N.3', 'Number of',
    'Number of laboratories' # Include this as it can sometimes be ambiguous
]
# Lowercase set for case-insensitive matching
N_PATTERNS_LOWER = {p.lower() for p in N_PATTERNS}

# Potential names for columns containing resistance metrics
RESISTANCE_PATTERNS = [
    "% ESBL", "%R", "%R .1", "%R .2", "%R .3",
    "%IR", "%IR .1", "%IR .2", "%IR .3",
    "% of total*", "% of total**"
]
# Lowercase set for case-insensitive matching
RESISTANCE_PATTERNS_LOWER = {p.lower() for p in RESISTANCE_PATTERNS}

# --- Final DataFrame Column Order ---
FINAL_COLUMN_ORDER = [
    "Microorganism", "Antibiotics", "Year", "Country", "N",
    "%R", "Fully susceptible", "%ESBL", "Non-susceptible",
    "Number of laboratories"
]