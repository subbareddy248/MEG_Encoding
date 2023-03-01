for feat in glove bert bertseq5 bertseq5rh bertseq1 postag deptags cm 
do 
    for i in $(seq 1 8);
    do
	    echo "Transferring subject-${i} (${feat})"
        rsync -avh "/media/nathan/easystore/meg_listening/meg_sub${i}_predictions/${feat}_s0_predictions" "reports/sub-${i}/"
    done
done

