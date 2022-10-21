python main.py \
--gpu_id 6 --manualSeed 4115 --cls_weight 0.01 --val_every 1 --preprocessing --cuda \
--image_embedding res101 --class_embedding att \
--nepoch 40 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset SUN --batch_size 64 \
--nz 102 --attSize 102 --resSize 2048 --lr 0.0001 --syn_num 400 --classifier_lr 0.001 \
--nclass_all 717 --outname sun --clb_time 10 --clb_weight 0.1 --clb_ratio 0.4 --neighbor_num 2



