Src="./valid-data/zh-en/zh.txt"
Tgt="./valid-data/zh-en/en.txt"
BpeSrc=$Src.bpe
Hyp="./model_output/500w-ft-zh-en.txt"
BpeHyp="./model_output/500w-ft-zh-en.txt.bpe"
Alignment="/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/alignments/zh-en/final_align.gdfa.valid"
python3 -m src.evaluation.eval_word_translation.alignment \
        --src $Src \
        --tgt $Tgt \
        --bped_src $BpeSrc \
        --bped_hyps $BpeHyp \
        --hyp $Hyp \
        --alignments $Alignment \
        --count_file /home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/cn-split-sen-zh-en-pretrain/zh.count.txt