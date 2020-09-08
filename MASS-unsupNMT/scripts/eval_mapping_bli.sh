export CUDA_VISIBLE_DEVICES="1"
python3 -m src.evaluation.eval_representation.eval_bli \
    --method "map" \
    --dict "/home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/BLI/zh-en/zh-en.txt.sim" \
    --save_path "./mapping.txt" \
    --src_embs "/home/data_ti5_d/zhouzh/low-resource-mt/tools/vecmap/zh.mapped" \
    --tgt_embs "/home/data_ti5_d/zhouzh/low-resource-mt/tools/vecmap/en.mapped" \
    --using_vocab True \
    --src_vocab "./word_vocab/zh.vocab" \
    --tgt_vocab "./word_vocab/en.vocab"
