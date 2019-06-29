#!/usr/bin/env bash

project_root='D:\\Exp_HZH\Anon'

:<<COMMENT
FOR DIGIT DATASETS:
    --source=svhn --target=mnist --source_loader=mcd --target_loader=mcd --num_classes=10 --all_use --batch_size=128 --image_size=256 --one_step --lr=0.001 --max_epoch=200 ----optimizer=adam --ensemble_alpha=0.8 --rampup_length=80 --weight_consistency_upper=1.0 --mixup_beta=0.5 --supplementary_info=""
    --source=synsig --target=gtsrb --source_loader=mcd --target_loader=mcd --num_classes=43 --all_use --batch_size=128 --image_size=32 --one_step --lr=0.0002 --max_epoch=200 --optimizer=adam --batch_interval=100 --ensemble_alpha=0.8 --rampup_length=10 --weight_consistency_upper=1.0 --mixup_beta=0.5 --supplementary_info=""
    --source=mnist --target=usps --source_loader=mcd --target_loader=mcd --num_classes=10 --all_use --batch_size=128 --image_size=32 --one_step --lr=0.0002 --max_epoch=200 --optimizer=adam --batch_interval=100 --ensemble_alpha=0.8 --rampup_length=10 --weight_consistency_upper=1.0 --mixup_beta=0.5 --supplementary_info=""
COMMENT

source_domain_list=('svhn' 'mnist' 'mnist' 'usps')
target_domain_list=('mnist' 'svhn' 'usps' 'mnist')
source_domain_classes=(10 10 10 10)

for index_source in `seq 0 3`
    do
        python main.py \
            --source=${source_domain_list[index_source]} \
            --target=${target_domain_list[index_source]} \
            --source_loader=mcd \
            --target_loader=mcd \
            --num_classes=${source_domain_classes[index_source]} \
            --all_use \
            --backbone=resnet18 \
            --batch_size=128 \
            --image_size=256 \
            --one_step \
            --lr=0.001 \
            --max_epoch=200 \
            --optimizer=adam \
            --ensemble_alpha=0.8 \
            --rampup_length=80 \
            --weight_consistency_upper=1.0 \
            --mixup_beta=0.5 \
            --supplementary_info="target domain mixup consistency loss"
    done

<<COMMENT
FOR OFFICE_31 DAATSETS:
    --source=svhn --target=mnist --source_loader=mcd --target_loader=mcd --num_classes=10 --all_use --batch_size=128 --image_size=32 --one_step --lr=0.0002 --max_epoch=200 --optimizer=adam --batch_interval=100 --ensemble_alpha=0.8 --rampup_length=10 --weight_consistency_upper=1.0 --mixup_beta=0.5 --supplementary_info=""
    --source=synsig --target=gtsrb --source_loader=mcd --target_loader=mcd --num_classes=43 --all_use --batch_size=128 --image_size=32 --one_step --lr=0.0002 --max_epoch=200 --optimizer=adam --batch_interval=100 --ensemble_alpha=0.8 --rampup_length=10 --weight_consistency_upper=1.0 --mixup_beta=0.5 --supplementary_info=""
    --source=mnist --target=usps --source_loader=mcd --target_loader=mcd --num_classes=10 --all_use --batch_size=128 --image_size=32 --one_step --lr=0.0002 --max_epoch=200 --optimizer=adam --batch_interval=100 --ensemble_alpha=0.8 --rampup_length=10 --weight_consistency_upper=1.0 --mixup_beta=0.5 --supplementary_info=""
COMMENT

source_domain_list=('A' 'W' 'D')
target_domain_list=('A' 'W' 'D')
source_domain_classes=(31 31 31)

for index_source in `seq 0 2`
    do
        for index_target in `seq 0 2`
            do
               if [${source_domain_list[index_source]} != ${source_domain_list[index_source]}]
               then
                    python main.py \
                        --source=${source_domain_list[index_source]} \
                        --target=${target_domain_list[index]} \
                        --source_loader=mcd \
                        --target_loader=mcd \
                        --num_classes=${source_domain_classes[index_source]} \
                        --all_use \
                        --backbone=resnet18 \
                        --batch_size=128 \
                        --image_size=256 \
                        --one_step \
                        --lr=0.001 \
                        --max_epoch=200 \
                        ----optimizer=adam \
                        --ensemble_alpha=0.8 \
                        --rampup_length=80 \
                        --weight_consistency_upper=1.0 \
                        --mixup_beta=0.5 \
                        --supplementary_info="target domain mixup consistency loss"
               fi
            done
    done
