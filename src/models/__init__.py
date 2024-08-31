# Controll all modules from here...

from ImageModelTraining import StartImageTraining
from src.models.TextModelTraining import StartTextTraining
from FeatureFusion import StartFusionTraining

# start image model training
StartImageTraining()

# start text model training
StartTextTraining()

# start fusion training
StartFusionTraining()