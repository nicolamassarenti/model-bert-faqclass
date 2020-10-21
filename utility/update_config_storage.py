import os
import argparse
import logging
from google.cloud import storage

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser("update_config_storage")
parser.add_argument("--bucket_name",
                    help="Name of the destination bucket.", type=str)
parser.add_argument("--bucket_folder",
                    help="Name of the destination folder.", type=str)
args = parser.parse_args()

bucket_name = args.bucket_name
bucket_folder = args.bucket_folder

local_files = [
    "/home/nicolamassarenti/Documents/Progetti/bert-faqclass/model-bert-faqclass/config/log_config.ini"
]


storage_client = storage.Client()
bucket = storage_client.get_bucket(bucket_name)

for file in local_files:
    path = os.path.dirname(file)
    filename = os.path.basename(file)
    bucket_destination = bucket_folder + "/" + filename

    # Double checking that the file exists
    if not os.path.isfile(file):
        raise Exception("{} is not a file".format(file))

    # Uploading file
    bucket.blob(blob_name=bucket_destination).upload_from_filename(filename=file)