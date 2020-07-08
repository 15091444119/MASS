DataPath="/home/zhouzh/data/tokenized_data/para/en-fr/"
Src=$DataPath/valid.en-fr.fr
Tgt=$DataPath/valid.en-fr.en
BpeSrc="/home/data_ti4_c/zhouzh/low-resource-mt/mass_baseline/MASS/MASS-unsupNMT/data/processed/en-fr/valid.en-fr.fr"
Hyp="/home/data_ti4_c/zhouzh/low-resource-mt/mass_baseline/MASS/MASS-unsupNMT/dumped/unsupMT_enfr/d5273d9dpu/hypotheses/hyp0.fr-en.valid.txt"
Alignment="/home/data_ti4_c/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/alignments/en-fr/final_align.gdfa.valid"
python3 -m src.evaluation.eval_word_translation.alignment \
	--src $Src \
	--tgt $Tgt \
	--bped_src $BpeSrc \
	--hyp $Hyp \
 	--alignments $Alignment
