export CUDA_VISIBLE_DEVICES="3"
Model="/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-ft-jointbpe-jointvocab/q6vn71z093/checkpoint.pth"
InputPath="./valid-data/zh-en/zh.txt.bpe"
OutputWithOutBpe="./model_output/500w-ft-zh-en.txt"
OutputPath="./model_output/500w-ft-zh-en.txt.bpe"
RefWithOutBpe="./valid-data/zh-en/en.txt"
SrcLang="zh"
TgtLang="en"

python3 ./translate.py \
  --exp_name "tmp_translate" \
  --model_path $Model \
  --src_lang $SrcLang \
  --tgt_lang $TgtLang \
  --batch_size 64 \
  --output_path $OutputPath  < $InputPath

cp $OutputPath $OutputWithOutBpe
sed -i -r "s/(@@ )|(@@ ?$)//g" $OutputWithOutBpe

./src/evaluation/multi-bleu.perl $RefWithOutBpe  < $OutputWithOutBpe > $OutputPath.bleuscore
