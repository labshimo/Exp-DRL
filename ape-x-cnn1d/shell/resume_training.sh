ROOT_DIR='I:/experiment/shimomura/dqn/data/default'
NUM_EOISODES=5000
Initial=1
cd ${ROOT_DIR}

while :
do  
    if [ -d ${Initial} ]; then
        Initial=$(($Initial+1))
    else
        break
    fi
done
echo "create directory ${Initial}"
mkdir ${Initial}

mv data run test actor.csv learner.csv ${Initial}
if [ -e test.txt ]; then
    echo "test.txt found."
    mv test.txt ${Initial}
else
    echo "test.txt NOT found."
fi
cp -r saved_networks ${Initial}

mkdir data run test
NUM_EOISODES=5000
python ../train.py --frame_width 3 --state_length 20 --n_actions 2 --train 1 --load 1 --num_episodes ${NUM_EOISODES}