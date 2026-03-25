#!/usr/bin/env bash
#
# Download the 82 Atlas test-set proteins from the ATLAS database.
#
# Usage:
#   bash scripts/download_atlas_test.sh <output_dir>
#
# Each protein is saved as:
#   <output_dir>/<name>/init.pdb

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <output_dir>"
    echo "  output_dir: directory to store downloaded proteins"
    exit 1
fi

OUTPUT_DIR="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CSV_FILE="${REPO_ROOT}/splits/atlas_test.csv"
BASE_URL="https://www.dsimb.inserm.fr/ATLAS/database/ATLAS"

if [ ! -f "$CSV_FILE" ]; then
    echo "ERROR: CSV file not found: $CSV_FILE"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

TOTAL=$(tail -n +2 "$CSV_FILE" | wc -l | tr -d ' ')
COUNT=0

tail -n +2 "$CSV_FILE" | cut -d',' -f1 | while read -r name; do
    COUNT=$((COUNT + 1))
    PROTEIN_DIR="${OUTPUT_DIR}/${name}"

    if [ -f "${PROTEIN_DIR}/init.pdb" ]; then
        echo "[${COUNT}/${TOTAL}] ${name} -- already exists, skipping"
        continue
    fi

    echo "[${COUNT}/${TOTAL}] Downloading ${name} ..."
    ZIP_FILE="${OUTPUT_DIR}/${name}_protein.zip"

    if ! wget -q -O "$ZIP_FILE" "${BASE_URL}/${name}/${name}_protein.zip"; then
        echo "  WARNING: failed to download ${name}, skipping"
        rm -f "$ZIP_FILE"
        continue
    fi

    mkdir -p "$PROTEIN_DIR"
    unzip -q -o "$ZIP_FILE" -d "$PROTEIN_DIR"

    if [ -f "${PROTEIN_DIR}/${name}.pdb" ]; then
        mv "${PROTEIN_DIR}/${name}.pdb" "${PROTEIN_DIR}/init.pdb"
    else
        echo "  WARNING: ${name}.pdb not found in archive, skipping"
        rm -rf "$PROTEIN_DIR"
        rm -f "$ZIP_FILE"
        continue
    fi

    # Clean up trajectory files and other extras
    rm -f "${PROTEIN_DIR}"/*.xtc "${PROTEIN_DIR}"/*.tpr "${PROTEIN_DIR}"/README.txt
    rm -f "$ZIP_FILE"

    echo "  Done -> ${PROTEIN_DIR}/init.pdb"
done

echo ""
echo "Download complete. Output directory: ${OUTPUT_DIR}"
