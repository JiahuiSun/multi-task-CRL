actor_path=$1
device=$2
task=$3

s1=1 1 1 0.8 1.2 20
s2=1 1 1 0.8 1.2 25
s3=1 1 0.8 1.0 1.2 20
s4=1.2 0.8 1 1 1 25
s5=0.8 1.2 1 1 1 15
s6=1 1 1 1 1 20
s7=0.5 1.5 1 1 1 20
s8=1 1 0.5 1 1.5 20
s9=1 1 1 0.5 1.5 15
s10=1.5 0.5 0.5 1.5 1 25

for setting in s1 s2 s3 s4 s5 s6 s7 s8 s9 s10
do
    echo setting: $setting
    python test.py --device $device --actor_path $actor_path --seed $seed --task $task --weight $setting
done