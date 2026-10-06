#!/bin/bash
# Compile paper6/main.tex with XeLaTeX (twice for stable cross-refs).
# Format: IEEE Transactions on Signal Processing submission —
# IEEEtran journal mode (two-column); IEEEtran.cls V1.8b is vendored
# in this directory from CTAN (BasicTeX lacks it).
# Usage: bash build.sh
set -euo pipefail
export PATH="/Library/TeX/texbin:$PATH"

TARGET=main.tex

cd "$(dirname "$0")"
xelatex -interaction=nonstopmode "$TARGET"
xelatex -interaction=nonstopmode "$TARGET"

# Clean intermediate files (keep .tex, .pdf, .bst, .cls, .sty, figures/)
rm -f *.aux *.log *.out *.toc *.bbl *.blg *.xdv *.spl
echo "OK -> $(pwd)/${TARGET%.tex}.pdf"
