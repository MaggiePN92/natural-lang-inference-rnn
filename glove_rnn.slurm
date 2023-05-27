#!/bin/bash
#SBATCH --job-name=in5550_assignment2_magnuspn_victoste
#SBATCH --account=ec30
#SBATCH --time=01:30:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=accel
#SBATCH --gpus=1

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
python3 download_glove.py
python3 train_rnns.py lstm_glove --rnn_type=lstm --pooling=max --pooling_multiplier=1 --save=True --path2embedding=/fp/projects01/ec30/IN5550/magnuspn/82/model.bin
