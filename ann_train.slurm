#!/bin/bash
#SBATCH --job-name=in5550_assignment2_magnuspn_victoste
#SBATCH --account=ec30
#SBATCH --time=10:30:00
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

# by default, pass on any remaining command-line options

echo "freeze false"

# raw
python3 train_ann.py --pooling=mean --hidden_size=512 --num_layers=2 --epochs=3 --lr=0.02 --freeze=false  --batch_size=1024 
python3 train_ann.py --pooling=sum --hidden_size=512 --num_layers=2 --epochs=3 --lr=0.02 --freeze=false --batch_size=1024 
python3 train_ann.py --pooling=max --hidden_size=512 --num_layers=2 --epochs=3 --lr=0.02 --freeze=false --batch_size=1024 

# lemmatized 
python3 train_ann.py --freeze=false --lr=0.02 --prep=lemmatized --path2embedding=/fp/projects01/ec30/magvic_large_files/modelbins/21/parameters.bin --fasttext=true --pooling=mean --hidden_size=512 --num_layers=2 --epochs=3  --batch_size=1024 
python3 train_ann.py --freeze=false --lr=0.02 --prep=lemmatized --path2embedding=/fp/projects01/ec30/magvic_large_files/modelbins/21/parameters.bin --fasttext=true --pooling=sum --hidden_size=512 --num_layers=2 --epochs=3  --batch_size=1024 
python3 train_ann.py --freeze=false --lr=0.02 --prep=lemmatized --path2embedding=/fp/projects01/ec30/magvic_large_files/modelbins/21/parameters.bin --fasttext=true --pooling=max --hidden_size=512 --num_layers=2 --epochs=3  --batch_size=1024 

# pos
python3 train_ann.py --freeze=false --prep=pos --lr=0.02 --path2embedding=/fp/projects01/ec30/models/static/200/model.bin --pooling=mean --hidden_size=512 --num_layers=2 --epochs=3  --batch_size=1024 
python3 train_ann.py --freeze=false --prep=pos --lr=0.02 --path2embedding=/fp/projects01/ec30/models/static/200/model.bin --pooling=sum --hidden_size=512 --num_layers=2 --epochs=3  --batch_size=1024 
python3 train_ann.py --freeze=false --prep=pos --lr=0.02 --path2embedding=/fp/projects01/ec30/models/static/200/model.bin --pooling=max --hidden_size=512 --num_layers=2 --epochs=3  --batch_size=1024 

# echo "freeze true"

# # raw
# python3 train_ann.py --pooling=mean --hidden_size=512 --num_layers=2 --epochs=10 --lr=0.02 --freeze=True
# python3 train_ann.py --pooling=sum --hidden_size=512 --num_layers=2 --epochs=10 --lr=0.02 --freeze=True
# python3 train_ann.py --pooling=max --hidden_size=512 --num_layers=2 --epochs=10 --lr=0.02 --freeze=True
# 
# # lemmatized 
# python3 train_ann.py --freeze=True --lr=0.02 --prep=lemmatized --path2embedding=/fp/projects01/ec30/magvic_large_files/modelbins/21/parameters.bin --fasttext=true --pooling=mean --hidden_size=512 --num_layers=2 --epochs=10
# python3 train_ann.py --freeze=True --lr=0.02 --prep=lemmatized --path2embedding=/fp/projects01/ec30/magvic_large_files/modelbins/21/parameters.bin --fasttext=true --pooling=sum --hidden_size=512 --num_layers=2 --epochs=10
# python3 train_ann.py --freeze=True --lr=0.02 --prep=lemmatized --path2embedding=/fp/projects01/ec30/magvic_large_files/modelbins/21/parameters.bin --fasttext=true --pooling=max --hidden_size=512 --num_layers=2 --epochs=10
# 
# # pos
# python3 train_ann.py --freeze=True --prep=pos --lr=0.02 --path2embedding=/fp/projects01/ec30/models/static/200/model.bin --pooling=mean --hidden_size=512 --num_layers=2 --epochs=10
# python3 train_ann.py --freeze=True --prep=pos --lr=0.02 --path2embedding=/fp/projects01/ec30/models/static/200/model.bin --pooling=sum --hidden_size=512 --num_layers=2 --epochs=10
# python3 train_ann.py --freeze=True --prep=pos --lr=0.02 --path2embedding=/fp/projects01/ec30/models/static/200/model.bin --pooling=max --hidden_size=512 --num_layers=2 --epochs=10
