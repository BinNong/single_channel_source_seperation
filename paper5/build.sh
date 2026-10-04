#!/bin/bash
# Compile paper5/main.tex with XeLaTeX (twice for stable cross-refs).
# Usage: bash build.sh   |   bash build.sh cover   |   bash build.sh letter
set -euo pipefail
export PATH="/Library/TeX/texbin:$PATH"

case "${1:-}" in
    cover)  TARGET=cover_letter.tex ;;
    letter) TARGET=letter.tex ;;     # WCL version (IEEEtran)
    *)      TARGET=main.tex ;;
esac

cd "$(dirname "$0")"
if [[ "${1:-}" == "letter" ]]; then
    # IEEEtran letter: pdflatex (native IEEE fonts; xelatex falls back to
    # Latin Modern because TU/ptm is missing from BasicTeX)
    pdflatex -interaction=nonstopmode "$TARGET"
    pdflatex -interaction=nonstopmode "$TARGET"
else
    xelatex -interaction=nonstopmode "$TARGET"
    xelatex -interaction=nonstopmode "$TARGET"
fi

# Clean intermediate files (keep .tex, .pdf, .bst, .cls, .sty, figures/)
rm -f *.aux *.log *.out *.toc *.bbl *.blg *.xdv *.spl
echo "OK -> $(pwd)/${TARGET%.tex}.pdf"
