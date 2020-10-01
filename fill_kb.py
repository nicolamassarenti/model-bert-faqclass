import json
import datetime
from datetime import timezone
import os
from trainer.settings import settings
from trainer.services.firestoreService import FirestoreService

if __name__ == "__main__":
    settings()

    KB_COLLECTION = os.getenv("FIRESTORE_COLLECTION_KB")
    KEYWORDS_COLLECTION = os.getenv("FIRESTORE_COLLECTION_KEYWORDS")

    service = FirestoreService()

    with open('data/kb.json') as f:
        data = json.load(f)

    # Removing all content of KB_COLLECTION and inserting new data
    service.empty_collection(KB_COLLECTION)

    for faq in data["KB"]:
        faq["UpdateDate"] = datetime.datetime.now(tz=timezone.utc).isoformat()
        service.add(KB_COLLECTION, faq)

    # Removing all content of KEYWORDS_COLLECTION and inserting new data
    service.empty_collection(KEYWORDS_COLLECTION)

    for keyword in data["Keywords"]:
        keyword["UpdateDate"] = datetime.datetime.now(tz=timezone.utc).isoformat()
        service.add(KEYWORDS_COLLECTION, keyword)

