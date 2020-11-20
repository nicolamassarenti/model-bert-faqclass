from enum import Enum
from bert_faqclass.configurations import config


class CollectionSpecs(Enum):
    def __init__(self, name, description):
        self.name = name
        self.description = description

class CollectionsData(CollectionSpecs, Enum):
    KNOWLEDGE_BASE = config.gcloud.collections.knwoledge_base.name, config.gcloud.collections.knowledge_base.description
    KEYWORDS = config.gcloud.collections.keywords.name, config.gcloud.collections.keywords.description

