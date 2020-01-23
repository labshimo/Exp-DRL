NUM_EOISODES=5000
python train.py --frame_width 3 --state_length 20 --n_actions 2 --train 1 --load 0 --num_episodes ${NUM_EOISODES} 