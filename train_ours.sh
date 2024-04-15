export CUDA_VISIBLE_DEVICES="1"
python3 train_ours_v2.py --dataset cifar100 --lbl-percent 10 --novel-percent 50 --arch resnet18 --num_protos 200 --num_concepts 20 --lr 0.5