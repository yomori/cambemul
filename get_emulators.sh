#!/usr/bin/env bash
#
# Download every emulator subfolder from a Google Drive parent folder into
# ./emulators/, keeping each emulator's name:
#
#     emulators/<emulator name>/...
#
# Usage:
#   pip install gdown
#   chmod +x get_emulators.sh
#   ./get_emulators.sh                      # uses FOLDER_URL below
#   ./get_emulators.sh "<other folder url>" # override the parent folder
#
set -euo pipefail

# Parent Drive folder that holds one subfolder per emulator:
FOLDER_URL="${1:-https://drive.google.com/drive/folders/168BuUQGPRl2xeOgyU05Fo-1KSSclsMwg}"
DEST="emulators"

if ! command -v gdown >/dev/null 2>&1; then
  echo "gdown is not installed. Install it with:  pip install gdown" >&2
  exit 1
fi

mkdir -p "$DEST"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo ">>> downloading $FOLDER_URL"
( cd "$tmp" && gdown --folder "$FOLDER_URL" )

# gdown wraps everything in a directory named after the Drive folder.
# Lift the emulator subfolders out of that wrapper into emulators/.
shopt -s dotglob nullglob
top=( "$tmp"/* )
if [[ ${#top[@]} -eq 1 && -d "${top[0]}" ]]; then
  rsync -a "${top[0]}"/ "$DEST"/
else
  rsync -a "$tmp"/ "$DEST"/
fi
shopt -u dotglob nullglob

echo
echo "Done. Emulators in $DEST/:"
ls -1 "$DEST"