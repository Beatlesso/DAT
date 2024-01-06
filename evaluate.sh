PORT=30001
GPU=$1
CFG=$2
CKPT=$3

torchrun --nproc_per_node $GPU --master_port $PORT main.py --cfg $CFG --data-path /mnt/yicong.luo/dataset/ImageNet-1k --eval --resume $CKPT