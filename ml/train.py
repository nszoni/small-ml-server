#  _         _     
# | |       | |    
# | |__   __| |___ 
# | '_ \ / _` / __|
# | |_) | (_| \__ \
# |_.__/ \__,_|___/ recommender-systems
                  
import time
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate
import argparse
import joblib

#########################################################
# TODO:
# 1. Add option to perform cross validation instead of full training
# 2. Output nicely formatted json in a specific folder of your choice with:
#       - training time
#       - number of lines trained on
#       - date
#       - cross validation metrics (when in cross validation mode)
# 3. Add arguments for user to specify SVD hyperparameters to use
# see: surprise.prediction_algorithms.matrix_factorization.SVD documentation
#########################################################

def main(nlines: int) -> int:
    algo = SVD()
    data = Dataset.load_builtin(f'ml-{nlines}k')
    trainset = data.build_full_trainset()
    print(f"Starting training {nlines}k lines.")
    start = time.perf_counter()
    algo.fit(trainset)
    end = time.perf_counter()
    print(f"Training done in {end - start:0.4f} seconds.")
    with open(f"ml/models/model.surprisemodel", "wb") as fout:
        joblib.dump(algo, fout)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nlines", type=int)
    args = parser.parse_args()
    exit(main(args.nlines, args.save_name))
