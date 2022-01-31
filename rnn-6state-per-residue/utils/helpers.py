import os

def getDatasetPath() -> str:
    if os.name == "nt":
        return "G:\\My Drive\\Files\\Projects\\University\\2021S\\PBL\\train_set.fasta"
    else:
        return "/projects/University/2021S/PBL/train_set.fasta"
