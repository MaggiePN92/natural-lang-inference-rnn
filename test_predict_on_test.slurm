#!/bin/bash
#SBATCH --job-name=in5550_assignment2_magnuspn_victoste
#SBATCH --account=ec30
#SBATCH --time=00:15:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=accel
#SBATCH --gpus=1


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

python3 predict_on_test.py /fp/projects01/ec30/magnuspn/in5550/sample.tsv.gz --device=cuda
