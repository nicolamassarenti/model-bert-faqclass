from enum import Enum
from bert_faqclass.configurations import config

class LocationSpecs(Enum):
    def __init__(self, bucket, folders):
        self.bucket = bucket
        self.folders = folders
        self.complete_path = "gs://{bucket}/{folders}".format(
            bucket=bucket,
            folders=folders
        )

class Location(LocationSpecs, Enum):
    CHECKPOINTS_LOCATION = config.gcloud.storage.model.checkpoints
    TENSORBOARD_LOCATION = config.gcloud.storage.model.tensorboard
    MODEL_SAVINGS = config.gcloud.storage.model.model_savings

print("A")
print(Location.CHECKPOINTS_LOCATION.complete_path)