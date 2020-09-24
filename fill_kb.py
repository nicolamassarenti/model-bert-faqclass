import json
import datetime
from datetime import timezone
import os
from settings import settings
from src.services.firestoreService import FirestoreService

if __name__ == "__main__":
    settings()
    service = FirestoreService()

    with open('data/kb.json') as f:
        data = json.load(f)

    for faq in data["KB"]:
        faq["UpdateDate"] = datetime.datetime.now(tz=timezone.utc).isoformat()
        service.add(os.getenv("FIRESTORE_COLLECTION_KB"), faq)
