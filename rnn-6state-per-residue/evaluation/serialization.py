import os
import pandas as pd
import pickle

filenames = {
    "final_metrics": "final_metrics_df.pickle",
    "final_training_metrics": "final_training_metrics_df.pickle",
    "cv_training_metrics": "cv_training_metrics_df.pickle",
    "cv_metrics": "cv_metrics_df.pickle",
    "final_metrics_per_protein": "final_metrics_per_protein_df.pickle",
    "cv_metrics_per_protein": "cv_metrics_per_protein_df.pickle"
}
directory = "G:\\My Drive\\Files\\Projects\\University\\2021S\\PBL\\Code\\evaluation\\data"

class Serializer:
    def __init__(self, obj_type: str):
        pass

    @staticmethod
    def save(obj: pd.DataFrame, obj_type: str) -> None:
        with open(os.path.join(directory, filenames[obj_type]), "wb+") as f:
            pickle.dump(obj, f)

    @staticmethod
    def load(obj_type: str) -> pd.DataFrame:
        with open(os.path.join(directory, filenames[obj_type]), "rb") as f:
            return pickle.load(f)
