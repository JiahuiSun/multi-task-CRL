actor_path=$1
device=$2
task=$3

for seed in 100 102 104 106 108
do
    for threshold in 0 5 25 27.5 30 32.5 35 37.5 55 100
    do
        python test.py --device $device --actor_path $actor_path --seed $seed --threshold $threshold --task $task
    done
done