# seq,base,target_name,seq_esm
t="base"
bs=1
torchrun --nproc_per_node=2 --nnodes=1 train.py --model-choice transformer --bar 1 --batch-size ${bs} --num-epoch 200 --data-type ${t} --data-path ${PWD} --save-directory ${PWD}/pretrain_${t}_cut --seq2vec-path ${PWD}/seq2vec.pkl