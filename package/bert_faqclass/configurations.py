import os
import yaml

from bert_faqclass import CONFIG_LOCATION


def _get_config_dict(name: str) -> dict:
	with open(os.path.join(CONFIG_LOCATION, "{name}.yml".format(name=name))) as f:
		configmap = yaml.load(f, Loader=yaml.SafeLoader)
	return configmap if configmap else {}


######################################################################################
#  Gcloud                             ################################################
######################################################################################
class LocationConfig:
	def __init__(self, config: dict):
		self.bucket = config["bucket"]
		self.folders = config["folders"]

class StorageModelConfig:
	def __init__(self, config: dict):
		self.checkpoints = LocationConfig(config["checkpoints"])
		self.tensorboard = LocationConfig(config["tensorboard"])
		self.model_savings = LocationConfig(config["model_savings"])

class StorageConfig:
	def __init__(self, config: dict):
		self.prefix = config["prefix"]
		self.model = StorageModelConfig(config)

class CollectionConfig:
	def __init__(self, config: dict):
		self.name = config["name"]
		self.description = config["description"]

class FirestoreCollectionsConfig:
	def __init__(self, config: dict):
		self.knowledge_base = CollectionConfig(config["knowledge_base"])
		self.keywords = CollectionConfig(config["keywords"])

class DatabaseConfig:
	def __init__(self, config : dict):
		self.type = config["type"]
		self.collections = FirestoreCollectionsConfig(config["collections"])

class GcloudConfig:
	def __init__(self, config: dict):
		self.project = config["project"]
		self.database = DatabaseConfig(config["database"])
		self.storage = StorageConfig(config["storage"])

######################################################################################
#  Model                              ################################################
######################################################################################
class BertConfig:
	def __init__(self, config: dict):
		self.url = config["url"]

class InputsConfig:
	def __init__(self, config: dict):
		self.max_sequence_length = config["max_sequence_length"]

class SplitConfig:
	def __init__(self, config: dict):
		self.train = config["train"]
		self.validation = config["validation"]
		self.test = config["test"]

class TrainingConfig:
	def __init__(self, config: dict):
		self.epochs = config["epochs"]
		self.batch_size = config["batch_size"]
		self.load_checkpoints = config["load_checkpoints"]
		self.is_tensorboard_enabled = config["is_tensorboard_enabled"]
		self.is_checkpoints_enabled = config["is_checkpoints_enabled"]
		self.split = SplitConfig(config["split"])


class ModelConfig:
	def __init__(self, config: dict):
		self.name = config["name"]
		self.version = config["version"]
		self.bert = BertConfig(config["bert"])
		self.inputs = InputsConfig(config["inputs"])
		self.training = TrainingConfig(config["training"])

######################################################################################
#  Config                             ################################################
######################################################################################
class Config:
	def __init__(self, config: dict):
		self.gcloud = GcloudConfig(config["gcloud"])
		self.model = ModelConfig(config["model"])


config = Config(_get_config_dict(name="default"))
