actor_path=$1
device=$2
task=$3
seed=$4

for setting in "1 1 1 0.8 1.2 20" "1 1 1 0.8 1.2 25" "1 1 0.8 1.0 1.2 20" "1.2 0.8 1 1 1 25" "0.8 1.2 1 1 1 15" "1 1 1 1 1 20" "0.5 1.5 1 1 1 20" "1 1 0.5 1 1.5 20" "1 1 1 0.5 1.5 15" "1.5 0.5 0.5 1.5 1 25"
do
    echo setting: $setting
    python test.py --device $device --actor_path $actor_path --seed $seed --task $task --weight $setting
done