#!/bin/bash

#SBATCH --job-name=IN5550
#SBATCH --account=ec30
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=8G

# Increase this number when you really need parallel computing
# (don't set it to more than 6 or 8 cores):
#SBATCH --cpus-per-task=8

source ${HOME}/.bashrc

# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset

module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-gensim/4.2.0-foss-2021a-Python-3.9.5

echo $SUBMITDIR

# We pass as arguments the path to the training corpus and (optionally) the number of CPU cores we want to use
python3 prep_data_for_embedding.py 

echo "Corpus made - creating word embedding."

python3 train_word2vec.py --corpus /fp/projects01/ec30/magnuspn/big_files/prepped_corpus.gz --cores $SLURM_CPUS_ON_NODE --vocab 100000
