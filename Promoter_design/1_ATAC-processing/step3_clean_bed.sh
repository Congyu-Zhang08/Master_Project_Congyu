clean_bed() {
    INPUT_NAME=$1
    echo "Processing ${INPUT_NAME}..."
    
    
    awk 'BEGIN {OFS="\t"} {
        if ($1 ~ /^chr[0-9X]+$/) { 
            print $0;
        }
    }' "${INPUT_NAME}_raw.bed" | \
    bedtools intersect -v -a - -b hg38-blacklist.v2.bed > "${INPUT_NAME}_clean.bed"

    echo "  -> Done. Remaining: $(wc -l < ${INPUT_NAME}_clean.bed) peaks."
}

clean_bed "SKNSH"
clean_bed "K562"
clean_bed "HepG2"