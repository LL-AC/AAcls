-- example:
torchrun --nproc_per_node 4 --master_port 12345  main.py --model simpleVit

torchrun --nproc_per_node 4 --master_port 12345  main.py --model simpleVit --epochs 50 --output_dir ./v1 --log_dir ./v1 --pin_memory --ctiterion mse

torchrun --nproc_per_node 4 --master_port 12345  main.py --model simpleVit --epochs 100 --output_dir ./v1 --log_dir ./v1 --pin_memory --ctiterion mse --resume 
./v1/checkpoint-49.pth
