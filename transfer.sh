for i in $(seq 1 8);
do echo $i; mkdir -p ./reports/sub-$i; rsync -avh /media/nathan/easystore/meg_listening/meg_sub$i*/glove* ./reports/sub-$i
done

