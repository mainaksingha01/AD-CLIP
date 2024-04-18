cd ..

DATA=Data
TRAINER=ADCLIP
BACKBONE=ViTB16

DATASET=$1
CFG=$2  # config file
NAME=$3 # job name
SEED=$4
# LOADEP=5


for SEED in 2
do
    MODEL_DIR=output/${DATASET}/${TRAINER}/ViT-L14/${CFG}/${NAME}/seed_${SEED}
    DIR=output/${DATASET}/${TRAINER}/${CFG}/${U}_${NAME}/test_target/seed_${SEED}
    if false; then
        echo "The results already exist in ${DIR}"
    else
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir ${MODEL_DIR} \
        #--load-epoch ${LOADEP}\
        --eval-only
done
