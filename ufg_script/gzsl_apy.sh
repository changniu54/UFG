python main.py \
--gpu_id 6 --manualSeed 9182 --cls_weight 0.01 --preprocessing \
--lr 0.00001 --cuda --image_embedding res101 --class_embedding att \
--nepoch 30 --syn_num 1800 --ngh 4096 --ndh 4096 --lambda1 10 \
--critic_iter 5 --nclass_all 32 --dataset APY --batch_size 64 --nz 64 --attSize 64 \
--resSize 2048 --outname apy --clb_time 20 --clb_weight 0.1 --clb_ratio 0.4 --neighbor_num 4

