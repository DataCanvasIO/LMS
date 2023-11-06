# prune model with structure

lms pruning structure --model_name bloom-560m --pruned_model_path bloom-560m_structure --nsamples 128 --device cpu --model_type bloom --block_attention_layer_start 1 --block_attention_layer_end 22 --block_mlp_layer_start 1 --block_mlp_layer_end 22
