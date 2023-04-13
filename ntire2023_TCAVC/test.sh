CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 ./stage1/test.py
chmod 777 -R ./stage1_test_results
python ./stage2/inference_colorvid.py
chmod 777 -R ./stage2_test_results