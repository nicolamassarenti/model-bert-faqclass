from enum import Enum

from bert_faqclass.connectors.gcloud.config import gcloud_config


class FirestoreCollections(Enum):
    KNOWLEDGE_BASE = gcloud_config.database.collections.knowledge_base.name
    KEYWORDS = gcloud_config.database.collections.keywords.name
