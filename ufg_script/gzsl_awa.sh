python main.py \
--manualSeed 9182 --cls_weight 0.01 --preprocessing --val_every 1 \
--lr 0.00001 --cuda --image_embedding res101 --class_embedding att \
--nepoch 30 --syn_num 1800 \
--ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --nclass_all 50 \
--dataset AWA1 --batch_size 64 --nz 85 --attSize 85 --resSize 2048 --outname awa \
--clb_time 10 --clb_weight 0.1 --clb_ratio 0.4 --neighbor_num 4 --gpu_id 5


