from bert_faqclass.configurations import config


class GcloudConfig:
    def __init__(self, conf: dict):
        self.project = conf["project"]
        self.database = DatabaseConfig(conf["database"])
        self.storage = StorageConfig(conf["storage"])


########################################################################################################################


class DatabaseConfig:
    def __init__(self, conf: dict):
        self.type = conf["type"]
        self.collections = CollectionsConfig(conf["collections"])


class CollectionsConfig:
    def __init__(self, conf: dict):
        self.knowledge_base = CollectionDetails(conf["knowledge_base"])
        self.keywords = CollectionDetails(conf["keywords"])


class CollectionDetails:
    def __init__(self, conf: dict):
        self.name = conf["name"]


########################################################################################################################


class StorageConfig:
    def __init__(self, conf: dict):
        self.prefix = conf["prefix"]
        self.locations = LocationsConfig(conf["locations"])


class LocationsConfig:
    def __init__(self, conf: dict):
        self.checkpoints = LocationDetails(conf["checkpoints"])
        self.model_savings = LocationDetails(conf["model_savings"])


class LocationDetails:
    def __init__(self, conf: dict):
        self.bucket = conf["bucket"]
        self.folders = conf["folders"]


########################################################################################################################

gcloud_config = GcloudConfig(config["gcloud"])
