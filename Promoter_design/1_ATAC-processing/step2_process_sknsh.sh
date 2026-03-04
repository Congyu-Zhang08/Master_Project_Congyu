#!/bin/bash

zcat GSE262189_SK.N.SH_atac_fragments.tsv.gz | awk '
BEGIN {
    while ((getline < "sknsh_barcodes_rep1.txt") > 0) rep1[$1] = 1;
    while ((getline < "sknsh_barcodes_rep2.txt") > 0) rep2[$1] = 1;
}
{
    if ($1 ~ /^#/) next;
    bc = $4;
    if (bc in rep1) print $1, $2, $3 > "sknsh_rep1.bed";
    else if (bc in rep2) print $1, $2, $3 > "sknsh_rep2.bed";
    if (bc in rep1 || bc in rep2) print $1, $2, $3 > "sknsh_pooled.bed";
}'

MACS_ARGS="-f BED -g hs --nomodel --shift -75 --extsize 150 --keep-dup all -q 0.01"

macs2 callpeak -t sknsh_pooled.bed -n SKNSH_pooled $MACS_ARGS
macs2 callpeak -t sknsh_rep1.bed -n SKNSH_rep1 $MACS_ARGS
macs2 callpeak -t sknsh_rep2.bed -n SKNSH_rep2 $MACS_ARGS

bedtools intersect -u -f 0.50 -a SKNSH_pooled_peaks.narrowPeak -b SKNSH_rep1_peaks.narrowPeak > temp_overlap.bed
bedtools intersect -u -f 0.50 -a temp_overlap.bed -b SKNSH_rep2_peaks.narrowPeak > SKNSH_raw.bed

rm temp_overlap.bed