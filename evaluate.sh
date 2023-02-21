docs=100
SUMMARY="/home/ubuntu/rupadhyay/CREDPASS/result/40_60_finetuned_CLEF_wa_1000_ret10.${docs}.summary"
printf "run\tqrels\tmeasure\ttopic\tscore\n" > $SUMMARY
RUN_FILE_PATH="/home/ubuntu/rupadhyay/CREDPASS/result/60_40_TREC_micro_wa_1000_ret10.csv"
#RUN_FILE_PATH="/home/ubuntu/rupadhyay/CREDPASS/trec2020_BM25.csv"
QRELS="/home/ubuntu/rupadhyay/CREDPASS"

#compatibility="/home/ricky/Documents/PhDproject/Project_folder/Compatibility/compatibility.py"

trec_eval -q -c -M ${docs} -m cam_map -R qrels_twoaspects $QRELS/trec_qrels_2aspects.csv $RUN_FILE_PATH
#| gawk '{print "'$RUN_NAME'" "\t" "2aspects.useful-credible" "\t" $1 "\t" $2 "\t" $3}' >> $SUMMARY

trec_eval -q -c -M ${docs} -m cam -R qrels_twoaspects $QRELS/trec_qrels_2aspects.csv $RUN_FILE_PATH
#| gawk '{print "'$RUN_NAME'" "\t" "2aspects.useful-credible" "\t" $1 "\t" $2 "\t" $3}' >> $SUMMARY

#$trec_eval -q -c -M ${docs} -m nwcs -R qrels_twoaspects $QRELS/misinfo-qrels.2aspects.useful-credible $RUN_FILE_PATH | gawk '{print "'$RUN_NAME'" "\t" "2aspects.useful-credible" "\t" $1 "\t" $2 "\t" $3}' >> $SUMMARY

#$trec_eval -q -c -M 10 -m nlre -R qrels_twoaspects $QRELS/misinfo-qrels.2aspects.useful-credible $RUN_FILE_PATH | gawk '{print "'$RUN_NAME'" "\t" "2aspects.useful-credible" "\t" $1 "\t" $2 "\t" $3}' >> $SUMMARY

#python3 $compatibility $QRELS/misinfo-qrels-graded.harmful-only $RUN_FILE_PATH | gawk '{print "'$RUN_NAME'" "\t" "graded.harmful-only" "\t" $1 "\t" $2 "\t" $3}' >> $SUMMARY
#python3 $compatibility $QRELS/misinfo-qrels-graded.helpful-only $RUN_FILE_PATH | gawk '{print "'$RUN_NAME'" "\t" "graded.helpful-only" "\t" $1 "\t" $2 "\t" $3}' >> $SUMMARY