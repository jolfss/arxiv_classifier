#!/home/pcgta/mambaforge/envs/arxiv/bin/python
# #!/home/sean/mambaforge/envs/arxiv/bin/python3
#----------------------------------#
#   Sean Brynjólfsson (smb459)     #
#   Deep Learning * Assignment 6   #
#----------------------------------#
"""Module string."""

#-------------#
#   imports   #
#-------------#

# local import
from typing import Dict
from core import *

#---------------------------#
#   dataset & dataloaders   #
#---------------------------#
train_x : pd.DataFrame = pd.read_csv('train_x.csv')
"""[title]: a string containing the title of a paper\n [abstract]: a string containing the abstract of a paper\n[id]: an integer storing the id of the paper"""

test_x : pd.DataFrame  = pd.read_csv('test_x.csv')
"""[title]: a string containing the title of a paper\n [abstract]: a string containing the abstract of a paper\n[id]: an integer storing the id of the paper"""

train_y : pd.DataFrame  = pd.read_csv('train_y.csv')
"""[id]: an integer storing the id of the paper\n[primary_subfield]: a string containing the abstract of a paper\n[all_subfields]: a string containing space-separated subfield names\n[label]: the numeric representation of the primary subfield\n[all_subfields_numeric]: a string of space-separated numeric representation of all_subfields (ordered?)"""

#--------------------#
#   categorization   #
#--------------------#
abbrev_to_info: Dict[str,tuple[int, str]] = {}
label_to_info: Dict[int, tuple[str,str]] = {}
"""Maps each arXiv category abbreviation to its numeric and string representation.
Example: "cs.AI" -> 120, "Artificial Intelligence" """
with open("categories.txt", 'r') as file:
    for line in file.readlines():
        abbrev, fullname_paren = line.split(" (")
        fullname = fullname_paren[:-1]
        abbrev_to_info[abbrev] = (-1, fullname)

    for row in train_y.itertuples():
        abbrev = row.primary_subfield
        label = row.label
        (_, fullname) = abbrev_to_info[abbrev]
        abbrev_to_info[abbrev] = (label, fullname)
        label_to_info[label] = (abbrev, fullname)
