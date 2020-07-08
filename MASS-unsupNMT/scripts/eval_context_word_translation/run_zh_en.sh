DataPath="/home/zhouzh/data/tokenized_data/para/zh-en"
Src=$DataPath/valid.en-zh.zh
Tgt=$DataPath/valid.en-zh.en
BpeSrc="/home/data_ti4_c/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/cn-split-sen-zh-en-pretrain/valid.en-zh.zh"
Hyp="/home/data_ti4_c/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-supervised/u7s91zmjk2/hypotheses/hyp30.zh-en.valid.txt"
Alignment="/home/data_ti4_c/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/alignments/zh-en/final_align.gdfa.valid"
python3 -m src.evaluation.eval_word_translation.alignment \
	--src $Src \
	--tgt $Tgt \
	--bped_src $BpeSrc \
	--hyp $Hyp \
 	--alignments $Alignment
