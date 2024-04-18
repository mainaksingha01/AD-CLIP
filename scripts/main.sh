cd ..

DATA=Data # change your data path here
TRAINER=ADCLIP
BACKBONE=ViTB16

DATASET=$1 # name of the dataset
CFG=$2  # config file
NAME=$3 # job name
SEED=$4

for SEED in 1 2 3 4 5
do
    DIR=output/${DATASET}/${TRAINER}/${BACKBONE}/${CFG}/${NAME}/seed_${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR}
    fi
done