DISCLAIMER = (
    "Research and educational use only. Similarity does not imply the same "
    "diagnosis or clinical outcome. Mock predictions are never clinical results."
)

DEMO_BANNER = "Demo data - mock predictions"

LABELS = ("normal", "erythema", "erosion", "ulcer", "other_unmapped")
TARGET_LABELS = ("normal", "erythema", "erosion", "ulcer")

# Kvasir-Capsule style label remapping (extend as dataset docs require)
KVASIR_LABEL_MAP = {
    "Normal": "normal",
    "normal": "normal",
    "Normal clean mucosa": "normal",
    "Erythema": "erythema",
    "erythema": "erythema",
    "Erosion": "erosion",
    "erosion": "erosion",
    "Ulcer": "ulcer",
    "ulcer": "ulcer",
    "Angiectasia": "other_unmapped",
    "Blood": "other_unmapped",
    "Foreign Body": "other_unmapped",
    "Lymphangiectasia": "other_unmapped",
    "Polyp": "other_unmapped",
    "Reduced Mucosal View": "other_unmapped",
}
