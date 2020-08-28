DataPath="./output/cn-zh-en-bpe-pretrain-word-pretrain-25"
Src=$DataPath/input.txt.restore_bpe
Tgt=$DataPath/reference.txt.restore_bpe
BpeSrc=$DataPath/input.txt
BpeHyp=$DataPath/hyp
Hyp=$DataPath/hyp.restore_bpe
Alignment="/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/alignments/zh-en/final_align.gdfa.valid"
python3 -m src.evaluation.eval_word_translation.alignment \
	--src $Src \
	--tgt $Tgt \
	--bped_src $BpeSrc \
        --bped_hyps $BpeHyp \
	--hyp $Hyp \
 	--alignments $Alignment
