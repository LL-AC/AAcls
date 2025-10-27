CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port 12346  main.py --epochs 2  --dataset yourdataset --data_path yourpath \
 --batch_size 32 --accum_iter 1 --log_dir ./loggerpath --output_dir ./loggerpath/model --pin_memory --model yourmodel --lr 1e-3 \
 --warmup_epochs 2 --weight_decay 0.005 --ctiterion ce \
 --evaluate --resume checkpoint-best.pth
