# Evaluates predictors with different training data sizes

dataset=$1
predictor=$2
n_seeds=20
kwargs=$3

for n_train in 24 48 72 96 120 144 168 192 216 240; do
    sbatch scripts/evaluate_predictor.sh $dataset $predictor $n_seeds $n_train "$kwargs"
done
