#!/bin/bash

docker run \
--rm \
--name yodj_wsdan \
-it \
--runtime=nvidia \
-e NVIDIA_VISIBLE_DEVICES=2 \
-v /home/drstrange/working/deep_learning/repos/yodj_models:/JCJ \
-v /home/storage_disk2/datasets/CUB_200_2011/CUB_200_2011/images:/JCJ/tf_2/finegrained/wsdan/data \
-v /home/drstrange/working/deep_learning/docker/wsdan/train_model/logs:/logs \
-v /home/drstrange/working/deep_learning/docker/wsdan/train_model/ckpt:/ckpt \
-v /home/drstrange/working/deep_learning/docker/wsdan/train.yml:/train.yml \
yodj/wsdan \
python ./tf_2/finegrained/wsdan/train.py --config-file-path /train.yml
