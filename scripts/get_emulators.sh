#!/usr/bin/env bash
# Download all precomputed cambemul emulators from Google Drive in one shot,
# instead of fetching each emulator directory by hand.
#
#   bash scripts/get_emulators.sh                 # -> ./emulators/
#   bash scripts/get_emulators.sh <FOLDER_ID> <dest>
#
# Requires gdown:  pip install gdown   (or:  pip install "cambemul[download]")
set -euo pipefail

# Google Drive folder holding all published emulators (each a subdir of emu_*.npz).
FOLDER_ID="${1:-<FOLDER_ID>}"
DEST="${2:-emulators}"

if ! command -v gdown >/dev/null 2>&1; then
  echo "error: gdown not found. Install it with:  pip install gdown" >&2
  exit 1
fi
if [ "$FOLDER_ID" = "<FOLDER_ID>" ]; then
  echo "error: set the Google Drive FOLDER_ID (edit this script or pass it as arg 1)" >&2
  exit 1
fi

mkdir -p "$DEST"
echo "Downloading precomputed emulators into $DEST/ ..."
gdown --folder "https://drive.google.com/drive/folders/${FOLDER_ID}" -O "$DEST"
echo "Done. Load one with:"
echo "  python -c \"import cambemul; cambemul.loademul('$DEST/<name>')\""
