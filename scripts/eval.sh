cd ..

DATA=data  # change your data path here
MODE=test

DATASET=$1 # dataset name; officehome, visda17, mini_domainnet
TRAINER=$2 # ADCLIPRN50, ADCLIPB16, ADCLIPL14
CFG=$3  # config file; rn50, vitB16, vitL14
# SEED=$4


for SEED in 1 2 3 4 5
do
    MODEL_DIR=output/${DATASET}/${MODE}/${TRAINER}/${CFG}/seed_${SEED}
    DIR=output/${DATASET}/${MODE}/${TRAINER}/${CFG}/seed_${SEED}
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
