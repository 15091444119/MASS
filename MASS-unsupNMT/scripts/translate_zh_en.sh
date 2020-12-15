export CUDA_VISIBLE_DEVICES="2"
Model=/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-ft-jointbpe-jointvocab/q6vn71z093/checkpoint.pth
Src="zh"
Tgt="en"

InputPath=/home/data_ti5_d/zhouzh/low-resource-mt/alignment_data/data/valid-data/zh-en/$Src.txt.bpe
RefWithOutBpe=/home/data_ti5_d/zhouzh/low-resource-mt/alignment_data/data/valid-data/zh-en/$Tgt.txt
OutputDir="/home/data_ti5_d/zhouzh/low-resource-mt/alignment_data/data/model_outputs/unmt-$Src-$Tgt"
OutputWithOutBpe="$OutputDir/$Src-$Tgt.txt"
OutputPath="$OutputDir/$Src-$Tgt.txt.bpe"


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