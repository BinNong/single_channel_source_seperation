#!/bin/bash
# Compile paper/main.tex with XeLaTeX (twice for stable cross-refs).
# Usage: bash build.sh   |   bash build.sh cover
set -euo pipefail
export PATH="/Library/TeX/texbin:$PATH"

if [[ "${1:-}" == "cover" ]]; then
    TARGET=cover_letter.tex
else
    TARGET=main.tex
fi

cd "$(dirname "$0")"
xelatex -interaction=nonstopmode "$TARGET"
xelatex -interaction=nonstopmode "$TARGET"

# Clean intermediate files (keep .tex, .pdf, .bst, .cls, .sty, figures/)
rm -f *.aux *.log *.out *.toc *.bbl *.blg *.xdv
echo "OK -> $(pwd)/${TARGET%.tex}.pdf"
