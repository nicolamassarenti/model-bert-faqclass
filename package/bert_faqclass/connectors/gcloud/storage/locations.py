from enum import Enum
from bert_faqclass.connectors.gcloud.config import gcloud_config


class StorageLocationsSpecs:
    def __init__(self, bucket, folders):
        self.bucket = bucket
        self.folders = folders
        self.complete_path = "gs://{bucket}/{folders}".format(
            bucket=bucket, folders=folders
        )

    def __str__(self):
        return self.complete_path


class StorageLocations(StorageLocationsSpecs, Enum):
    CHECKPOINTS = (
        gcloud_config.storage.locations.checkpoints.bucket,
        gcloud_config.storage.locations.checkpoints.folders,
    )
    TENSORBOARD = (
        gcloud_config.storage.locations.tensorboard.bucket,
        gcloud_config.storage.locations.tensorboard.folders,
    )
    MODEL = (
        gcloud_config.storage.locations.model_savings.bucket,
        gcloud_config.storage.locations.model_savings.folders,
    )
