# data preprocess
`python preprocess.py -i /public/home/zhangjie/Projects/MMP/pot_clm/CLModel_v1/FinetuningData/AIXB-3/AIXB-3_AMPK/AIXB-3_AMPK_MMP.csv -d 1`

```shell
python preprocess.py -d 1 -i /home/yichao/zhilian/GenAICode/Data/MMPFinised/BindingDB_All_202407_1k/BindingDB_All_202407_1k_MMP.csv
```

# pretrian
`python train.py --model-choice transformer  --data-path  PretrainWork/ChEMBL32_Data   --save-directory PretrainWork/pretrain_chembl32`

```shell
python train.py --model-choice transformer --num-epoch 200  --data-path  /home/yichao/zhilian/GenAICode/CLModel_v2_zl  --save-directory /home/yichao/zhilian/GenAICode/CLModel_v2_zl/pretrain_v2
```
temp
```shell
python train.py --model-choice transformer  --data-path  /home/yichao/zhilian/GenAICode/Data/MMPFinised/BindingDB_All_202407_1k  --save-directory /home/yichao/zhilian/GenAICode/CLModel_v2_zl/pretrain_temp
```
torchrun
```shell
torchrun --nproc_per_node=2 --nnodes=1 train.py --model-choice transformer --num-epoch 200 --data-type base --data-path /home/yichao/zhilian/GenAICode/new_paper_code --save-directory /home/yichao/zhilian/GenAICode/new_paper_code/pretrain_tar
```

# fine tuning
`python train.py --model-choice transformer  --data-path /public/home/zhangjie/Projects/MMP/pot_clm/CLModel_v1/FinetuningData/AIXB-3/AIXB-3_JAK1  --save-directory FinetunedModels/finetune-AIXB3-JAK1 --starting-epoch 43 --pretrain-path /public/home/zhangjie/Projects/MMP/pot_clm/CLModel_v1/Pretrain/train  --vocab-path /public/home/zhangjie/Projects/MMP/pot_clm/CLModel_v1/TrainData_ChEMBL17`

# generate
` python generate.py --model-choice transformer$$  --data-path  /public/home/zhangjie/Projects/MMP/pot_clm/CLModel_v1/test_data   --test-file-name  test  --model-path /public/home/zhangjie/Projects/MMP/pot_clm/CLModel_v1/pretrain_test/checkpoint  --epoch 40    --vocab-path /public/home/zhangjie/Projects/MMP/pot_clm/CLModel_v1/TrainData_ChEMBL17  --save-directory FinetunedModels/finetune-AIXB3-JAK1 `

```shell
python generate.py --model-choice transformer  --data-path  /home/yichao/zhilian/GenAICode/CLModel_v2_zl --test-file-name  test_100  --model-path /home/yichao/zhilian/GenAICode/CLModel_v2_zl/pretrain_v1/checkpoint  --epoch 4 --vocab-path /home/yichao/zhilian/GenAICode/CLModel_v2_zl  --save-directory /home/yichao/zhilian/GenAICode/CLModel_v2_zl/generate_temp
```