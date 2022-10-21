python main.py \
--gpu_id 6 --manualSeed 3483 --cls_weight 0.01 \
--preprocessing --cuda --image_embedding res101 --class_embedding att \
--nepoch 50 --ngh 4096 --ndh 4096 \
--lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataset CUB \
--nclass_all 200 --batch_size 64 --nz 312 --attSize 312 --resSize 2048 --syn_num 300 \
--outname cub --clb_time 10 --clb_weight 0.1 --clb_ratio 0.4 --neighbor_num 3
