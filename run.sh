CUDA_VISIBLE_DEVICES=2,3 mpirun -np 2 python3.7 train.py ~/datasets/cifar cifar10  --learning_rate 0.05 -b 128
