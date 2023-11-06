#purne model with sparse

lms pruning sparse --model_name alaya065-ep2-fp16 --pruned_model_path alaya_sparsed --nsamples 128 --device cpu --layer_name_start blocks.3 --layer_name_stop blocks.4 
