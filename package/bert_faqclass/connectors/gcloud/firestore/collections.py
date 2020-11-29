from enum import Enum
from bert_faqclass.connectors.gcloud.config import gcloud_config


class CollectionSpec:
    def __init__(self, name, description):
        self.name = name
        self.description = description


class CollectionsConfig(CollectionSpec, Enum):
    KNOWLEDGE_BASE = gcloud_config.database.collections.knowledge_base.name, \
                     gcloud_config.database.collections.knowledge_base.description
    KEYWORDS = gcloud_config.database.collections.keywords.name, gcloud_config.database.collections.keywords.description
