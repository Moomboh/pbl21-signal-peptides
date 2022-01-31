from typing import Dict

from .Model import Model
from .BaseModel import BaseModel
from .AAindexModel import AAindexModel
from .BlosumModel import BlosumModel
from .CombinedModel import CombinedModel

models: Dict[str, Model] = {
    'Base': BaseModel,
    'AAindex': AAindexModel,
    'Blosum': BlosumModel,
    'Combined': CombinedModel,
}
