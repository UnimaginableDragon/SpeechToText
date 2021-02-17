python data.py ./Data/data.zip ./Data/Train/Best_hyperparameter_80_percent/ ./Data/Validation/Validation_10_percent/ ./Data/Test/Test_10_percent/ ./Data/Train/Under_10_min_training/ ./Data/Train/Under_90_min_tuning/ ./Data/Validation/3_samples/

python tune.py ./Data/Train/Under_90_min_tuning/data.zip ./Data/Validation/Validation_10_percent/data.zip ./tuning_results.txt ./hyperparameter.txt

python train.py ./Data/Train/Best_hyperparameter_80_percent/data.zip ./hyperparameter.txt ./model.h5


python test.py ./Data/Test/Test_10_percent/data.zip ./model.h5