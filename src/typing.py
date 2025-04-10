from typing import Tuple, Union

import torch
from numpy.typing import NDArray
from rdkit import Chem
from typing_extensions import TypeAlias

Device: TypeAlias = torch.device
Input: TypeAlias = Union[Tuple[NDArray, NDArray], NDArray]
Target: TypeAlias = NDArray
Mol: TypeAlias = Chem.rdchem.Mol
