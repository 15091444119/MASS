export CUDA_VISIBLE_DEVICES="3"
FtZh=/home/data_ti5_d/zhouzh/low-resource-mt/subword-combiner/MASS-unsupNMT/dumped/ft-CHAR-gru-zh/4igbacw6ao/best-valid_whole_combiner_acc_top1_acc.pth
FtEn=/home/data_ti5_d/zhouzh/low-resource-mt/subword-combiner/MASS-unsupNMT/dumped/ft-CHAR-gru-en/dmvy6f3xrt/best-valid_whole_combiner_acc_top1_acc.pth
FtMass=/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-ft-jointbpe-jointvocab/q6vn71z093/checkpoint.pth
PreMass=/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-pretrain-jointbpe-jointvocab/1jnrqg51bz/periodic-100.pth
python3 -m src.evaluation.eval_combiner_bli \
--mass_model $PreMass \
--src_combiner_model  /home/data_ti5_d/zhouzh/low-resource-mt/subword-combiner/MASS-unsupNMT/dumped/pretrain-CHAR-gru-average-zh/l0rnq24xxt/best-valid_whole_combiner_acc_top1_acc.pth \
--tgt_combiner_model /home/data_ti5_d/zhouzh/low-resource-mt/subword-combiner/MASS-unsupNMT/dumped/pretrain-CHAR-gru-average-en/xvytlvwj53/best-valid_whole_combiner_acc_top1_acc.pth \
--src_bped_word ./word_vocab/zh.vocab.bpe \
--tgt_bped_word ./word_vocab/en.vocab.bpe \
--dict_path /home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/BLI/zh-en/zh-en.txt.sim \
--save_path ~/new_eval