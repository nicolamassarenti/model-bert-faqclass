import logging

from firebase_admin import firestore
import firebase_admin


firebase_admin.initialize_app()
connector = firestore.client()
logging.info("Firestore connector created")