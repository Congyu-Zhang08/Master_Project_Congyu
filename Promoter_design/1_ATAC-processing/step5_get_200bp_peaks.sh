#!/bin/bash

process_final() {
    CELL=$1
    TOP_N=12000
    
    grep -v "^#" "${CELL}_specific.bed" | \
    sort -k9,9nr | \
    head -n "$TOP_N" | \
    awk -v tag="$CELL" 'BEGIN{OFS="\t"} {
        summit_abs = $2 + $10; 
        start = summit_abs - 100; 
        end = summit_abs + 100;
        if (start >= 0) 
            print $1, start, end, tag;
    }' > "${CELL}_final_200bp.bed"
}

process_final "SKNSH"
process_final "K562"
process_final "HepG2"

cat *_final_200bp.bed > all_cells_training.bed
