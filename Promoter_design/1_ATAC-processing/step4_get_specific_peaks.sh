#!/bin/bash

# SKNSH
cat K562_clean.bed HepG2_clean.bed | sort -k1,1 -k2,2n | bedtools merge -i - > bg_sknsh.bed
bedtools intersect -v -a SKNSH_clean.bed -b bg_sknsh.bed > SKNSH_specific.bed
echo "SKNSH specific peaks: $(wc -l < SKNSH_specific.bed)"

# K562
cat SKNSH_clean.bed HepG2_clean.bed | sort -k1,1 -k2,2n | bedtools merge -i - > bg_k562.bed
bedtools intersect -v -a K562_clean.bed -b bg_k562.bed > K562_specific.bed
echo "K562 specific peaks: $(wc -l < K562_specific.bed)"

# HepG2
cat SKNSH_clean.bed K562_clean.bed | sort -k1,1 -k2,2n | bedtools merge -i - > bg_hepg2.bed
bedtools intersect -v -a HepG2_clean.bed -b bg_hepg2.bed > HepG2_specific.bed
echo "HepG2 specific peaks: $(wc -l < HepG2_specific.bed)"

rm bg_*.bed