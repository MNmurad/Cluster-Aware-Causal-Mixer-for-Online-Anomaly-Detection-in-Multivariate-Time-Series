if [ ! -d "./logs/Results" ]; then
    mkdir ./logs/Results
fi
if [ ! -d "./logs/Results/err" ]; then
    mkdir ./logs/Results/err
fi

export CUDA_VISIBLE_DEVICES=0


# General
model_name=CCM_TAD
data=SWAT
seed=42
timestamp=$(date +"%Y%m%d_%H%M%S") # single log per entity 

# to make the training faster, I will train the model using point-based anom detection. 
# Then, I will use the checkpoint from point based to get the result for sequence based anomaly detection

# Point based anomaly detection
target_optimization=pf1
checkpoint_load=0
python -u run_anomaly.py \
	--task_name anomaly_detection \
	--model $model_name \
	--data $data \
	--seed $seed \
	--target_optimization $target_optimization \
	--checkpoint_load $checkpoint_load \
	> logs/Results/${data}_ablation${abID}_${target_optimization}_${timestamp}.log \
	2> logs/Results/err/${data}_${target_optimization}_${timestamp}.err &
wait


# Proposed Sequence based anomaly detecton
target_optimization=sf1
checkpoint_load=1
python -u run_anomaly.py \
	--task_name anomaly_detection \
	--model $model_name \
	--data $data \
	--seed $seed \
	--target_optimization $target_optimization \
	--checkpoint_load $checkpoint_load \
	> logs/Results/${data}_ablation${abID}_${target_optimization}_${timestamp}.log \
	2> logs/Results/err/${data}_${target_optimization}_${timestamp}.err &
wait
