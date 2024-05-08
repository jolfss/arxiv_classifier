#!/home/pcgta/mambaforge/envs/arxiv/bin/python
# #!/home/sean/mambaforge/envs/arxiv/bin/python3
#----------------------------------#
#   Sean Brynjólfsson (smb459)     #
#   Deep Learning * Assignment 6   #
#----------------------------------#
"""This file contains the training script."""

# local
from core import *  
from core import _model # NOTE: Make sure core's model is set to the appropriate specter(2(2023)) model
from data import *

x = to_raw_embeddings(train_x)
