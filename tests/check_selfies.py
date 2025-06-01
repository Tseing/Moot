import sys

from rdkit import Chem

sys.path.append("..")
import src.selfies as sf

s = r"*C(=O)N[C@H]1CC[C@H](CCN2CCN(c3cccc4c3OCO4)CC2)CC1"
s = s.replace("*", "[*]")
print(s)
encoded = sf.encoder(s)
print("->", encoded)
decoded = sf.decoder(encoded)
print("->", decoded)

assert Chem.CanonSmiles(decoded) ==Chem.CanonSmiles(s)