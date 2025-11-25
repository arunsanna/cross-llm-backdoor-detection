#!/bin/bash
# LaTeX Compilation Script
# Compiles main.tex with proper bibliography processing

set -e  # Exit on error

# Update PATH to include LaTeX
eval "$(/usr/libexec/path_helper)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== LaTeX Compilation Script ===${NC}"
echo ""

# Check if main.tex exists
if [ ! -f "main.tex" ]; then
    echo -e "${RED}Error: main.tex not found!${NC}"
    echo "Please run this script from the paper/ directory"
    exit 1
fi

# Check if LaTeX is installed
if ! command -v pdflatex &> /dev/null; then
    echo -e "${RED}Error: pdflatex not found!${NC}"
    echo "Please install MacTeX: brew install --cask mactex"
    exit 1
fi

echo -e "${YELLOW}Step 1/5: First pdflatex pass (generating aux files)...${NC}"
pdflatex -interaction=nonstopmode -halt-on-error main.tex || {
    echo -e "${RED}First pdflatex pass failed!${NC}"
    echo "Check main.log for errors"
    exit 1
}

echo -e "${YELLOW}Step 2/5: Running bibtex (processing bibliography)...${NC}"
bibtex main || {
    echo -e "${RED}BibTeX failed!${NC}"
    echo "Check main.blg for errors"
    exit 1
}

echo -e "${YELLOW}Step 3/5: Second pdflatex pass (incorporating bibliography)...${NC}"
pdflatex -interaction=nonstopmode -halt-on-error main.tex || {
    echo -e "${RED}Second pdflatex pass failed!${NC}"
    exit 1
}

echo -e "${YELLOW}Step 4/5: Third pdflatex pass (resolving references)...${NC}"
pdflatex -interaction=nonstopmode -halt-on-error main.tex || {
    echo -e "${RED}Third pdflatex pass failed!${NC}"
    exit 1
}

echo -e "${YELLOW}Step 5/5: Cleaning up auxiliary files...${NC}"
rm -f *.aux *.log *.bbl *.blg *.out *.toc *.lof *.lot

echo ""
echo -e "${GREEN}=== Compilation Successful! ===${NC}"
echo -e "Output: ${GREEN}main.pdf${NC}"
echo ""

# Check PDF size
if [ -f "main.pdf" ]; then
    SIZE=$(du -h main.pdf | cut -f1)
    PAGES=$(pdfinfo main.pdf 2>/dev/null | grep Pages | awk '{print $2}' || echo "unknown")
    echo -e "PDF Size: ${GREEN}${SIZE}${NC}"
    echo -e "Pages: ${GREEN}${PAGES}${NC}"
fi

echo ""
echo "To view: open main.pdf"
