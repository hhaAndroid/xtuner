path=$1

cd /mnt/hwfile/xtuner/huanghaian/model/Mini-InternVL-Chat-4B-V1-5/
cp modeling_* $path
cp conversation.py $path
cp preprocessor_config.json $path
cp /mnt/petrelfs/huanghaian/code/xtuner/my_llava/modeling_phi3.py $path
