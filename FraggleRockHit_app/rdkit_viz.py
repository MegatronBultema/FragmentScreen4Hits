#http://rdkit.blogspot.com/2016/02/morgan-fingerprint-bit-statistics.html
import numpy
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import AllChem as Chem
from rdkit import DataStructs
import pickle
from collections import Counter
from rdkit import rdBase
print(rdBase.rdkitVersion)
#from IPython.display import SVG
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
