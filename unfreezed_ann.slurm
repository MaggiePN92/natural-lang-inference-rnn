#!/bin/bash
#SBATCH --job-name=embedding_search
#SBATCH --account=ec30
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16G
#
# by default, request two cores (NumPy may just know how to take
# advantage of them; for larger computations, maybe use between
# six and ten; at some point, we will look at how to run on gpus
#
#SBATCH --cpus-per-task=8

# NB: this script should be run with "sbatch ...
# See https://www.uio.no/english/services/it/research/platforms/edu-research/help/fox/jobs/submitting.md

source ${HOME}/.bashrc

# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset

# the important bit: unload all current modules (just in case) and load only the necessary ones
module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-nlptools/2022.01-foss-2021a-Python-3.9.5
module load nlpl-pytorch/1.11.0-foss-2021a-cuda-11.3.1-Python-3.9.5
module load nlpl-gensim/4.2.0-foss-2021a-Python-3.9.5

# print information (optional)
echo "submission directory: ${SUBMITDIR}"


# raw
python3 train_ann.py --hidden_size=512 --num_layers=2 --epochs=5 --lr=0.02 --batch_size=512 --pooling=sum --freeze=false

