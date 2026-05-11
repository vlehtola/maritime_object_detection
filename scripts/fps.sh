config_file=$1
ckp=$2

source /home/tinsae/miniconda3/bin/activate /home/tinsae/miniconda3/envs/mtp
export PYTHONPATH=$PYTHONPATH:/home/tinsae/Desktop/MTP/Codebase/marine_det
/home/tinsae/miniconda3/envs/mtp/bin/python3 third_party/mmdet/tools/analysis_tools/benchmark.py $config_file --checkpoint $ckp --task inference

# source /home/ubuntu/fugro.madlab.marine_det/marine_det/myenv/bin/activate
# export PYTHONPATH=$(pwd)
# python third_party/mmdet/tools/analysis_tools/benchmark.py $config_file --checkpoint $ckp --task inference
