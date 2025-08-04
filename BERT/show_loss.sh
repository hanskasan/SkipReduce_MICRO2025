###### EVALUATION LOSS #####
# grep 'eval_accuracy' aot_reports/baseline_seed1.report | awk '{gsub(",", "", $4); print $4}'
# grep 'eval_accuracy' aot_reports/skipreduce_50_seed2.report | awk '{gsub(",", "", $4); print $4}'
grep 'eval_accuracy' aot_reports/powerSGD_rank4_seed2.report | awk '{gsub(",", "", $4); print $4}'
# grep 'eval_accuracy' aot_reports/top1_seed0_dataseed4.report | awk '{gsub(",", "", $4); print $4}'

