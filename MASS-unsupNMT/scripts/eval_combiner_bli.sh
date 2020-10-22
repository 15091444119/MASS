export CUDA_VISIBLE_DEVICES="6"
python3 -m src.evaluation.eval_combiner_bli \
  --mass_model /home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-pretrain-jointbpe-jointvocab/1jnrqg51bz/periodic-100.pth \
  --src_combiner_model ./dumped/pretrain100-CHAR-gru-zh/dvptir0l0h/best-valid_whole_combiner_acc_top1_acc.pth \
  --tgt_combiner_model ./dumped/pretrain100-CHAR-gru-en/6uox99arpd/best-valid_whole_combiner_acc_top1_acc.pth \
  --src_bped_word ./word_vocab/zh.vocab.bpe \
  --tgt_bped_word ./word_vocab/en.vocab.bpe \
  --dict_path /home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/BLI/zh-en/zh-en.txt.sim \
  --save_path ./pretrain100-gru-bli.txt