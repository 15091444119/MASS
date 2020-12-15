export CUDA_VISIBLE_DEVICES="2"
Model=/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/en-fr-500w-split-ft/qmb5bz8o3v/checkpoint.pth

Src="en"
Tgt="fr"
InputPath=/home/data_ti5_d/zhouzh/low-resource-mt/alignment_data/data/valid-data/fr-en/valid.en-fr.$Src
OutputDir="/home/data_ti5_d/zhouzh/low-resource-mt/alignment_data/data/model_outputs/unmt-$Src-$Tgt"
OutputWithOutBpe="$OutputDir/$Src-$Tgt.txt"
OutputPath="$OutputDir/$Src-$Tgt.txt.bpe"
RefWithOutBpe=/home/data_ti5_d/zhouzh/low-resource-mt/alignment_data/data/valid-data/fr-en/valid.en-fr.$Tgt.nobpe


rm -r $OutputDir
mkdir -p $OutputDir

python3 ./translate.py \
  --exp_name "tmp_translate" \
  --model_path $Model \
  --src_lang $Src \
  --tgt_lang $Tgt \
  --batch_size 64 \
  --output_path $OutputPath  < $InputPath

cp $OutputPath $OutputWithOutBpe
sed -i -r "s/(@@ )|(@@ ?$)//g" $OutputWithOutBpe

./src/evaluation/multi-bleu.perl $RefWithOutBpe  < $OutputWithOutBpe > $OutputPath.bleuscore

cat $OutputPath.bleuscore