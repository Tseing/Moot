import sys

sys.path.append("..")


from src.tokenizer import AtomTokenizer, ProteinTokenizer, SmilesTokenizer, share_vocab

if __name__ == "__main__":
    smiles = "CC1(N)Cc2ccccc21"
    smiles_tokenizer = SmilesTokenizer()
    smiles_tokenizer.load_word_table("../data/all/smiles_word_table.yaml")
    tokenized_smiles = smiles_tokenizer.tokenize(smiles)
    print(tokenized_smiles)
    print(smiles_tokenizer.convert_ids2tokens(tokenized_smiles))

    raw_protein = "METLCLRASFWLALVGCVISDNPERYSTNLSNHVDDFTTFRGTELSFLVTTHQPTNLVLPSNGSMHNYCPQQTKITSAFKYINTVISCTIFIVGMVGNATLLRIIYQNKCMRNGPNALIASLALGDLIYVVIDLPINVFKLLAGRWPFDHNDFGVFLCKLFPFLQKSSVGITVLNLCALSVDRYRAVASWSRVQGIGIPLVTAIEIVSIWILSFILAIPEAIGFVMVPFEYRGEQHKTCMLNATSKFMEFYQDVKDWWLFGFYFCMPLVCTAIFYTLMTCEMLNRRNGSLRIALSEHLKQRREVAKTVFCLVVIFALCWFPLHLSRILKKTVYNEMDKNRCELLSFLLLMDYIGINLATMNSCINPIALYFVSKKFKNCFQSCLCCCCYQSKSLMTSVPMNGTSIQWKNHDQNNHNTDRSSHKDSMN"
    protein = "".join([f"-{symbol}" for symbol in raw_protein])
    protein_tokenizer = ProteinTokenizer()
    tokenized_protein = protein_tokenizer.tokenize(protein)
    print(tokenized_protein)
    print(protein_tokenizer.convert_ids2tokens(tokenized_protein))

    protein_tokenizer, smiles_tokenizer = share_vocab(protein_tokenizer, smiles_tokenizer)
    print(smiles_tokenizer.word_table)
    print(protein_tokenizer.word_table)

    tokenized_smiles = smiles_tokenizer.tokenize(smiles)
    print(tokenized_smiles)
    print(smiles_tokenizer.convert_ids2tokens(tokenized_smiles))

    tokenized_protein = protein_tokenizer.tokenize(protein)
    print(tokenized_protein)
    print(protein_tokenizer.convert_ids2tokens(tokenized_protein))

    atoms = "[Cr]{unk}[C][C][C][C][N][*][C][C][C][C][C][C][O][N][C][C][C][C][C][C][N][C][C][O][C][C][N][C][C][C][N][C][C][C][C][C][C][C][C][C][C][C]"
    atom_tokenizer = AtomTokenizer()
    print(atom_tokenizer.tokenize(atoms))
