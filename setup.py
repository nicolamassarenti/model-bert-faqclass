from setuptools import find_packages
from setuptools import setup
import os


# # Read requirements
# requirements_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")
# with open(requirements_file, "r") as f:
#     requirements = f.read().splitlines()
requirements = ['absl-py==0.10.0', 'astunparse==1.6.3', 'attrs==20.2.0', 'CacheControl==0.12.6', 'cachetools==4.1.1', 'certifi==2020.6.20', 'cffi==1.14.2', 'chardet==3.0.4', 'cloudpickle==1.6.0', 'colorama==0.4.3', 'Cython==0.29.21', 'dill==0.3.2', 'docker==4.3.1', 'erlastic==2.0.0', 'firebase-admin==4.4.0', 'future==0.18.2', 'gast==0.3.3', 'google-api-core==1.22.2', 'google-api-python-client==1.12.1', 'google-auth==1.21.1', 'google-auth-httplib2==0.0.4', 'google-auth-oauthlib==0.4.1', 'google-cloud-core==1.4.1', 'google-cloud-firestore==1.9.0', 'google-cloud-storage==1.31.0', 'google-crc32c==1.0.0', 'google-pasta==0.2.0', 'google-resumable-media==1.0.0', 'googleapis-common-protos==1.52.0', 'grpcio==1.32.0', 'h5py==2.10.0', 'httplib2==0.18.1', 'idna==2.10', 'joblib==0.16.0', 'Keras-Preprocessing==1.1.2', 'keras-tuner==1.0.1', 'Markdown==3.2.2', 'msgpack==1.0.0', 'numpy==1.18.5', 'oauthlib==3.1.0', 'opt-einsum==3.3.0', 'promise==2.3', 'protobuf==3.13.0', 'pyasn1==0.4.8', 'pyasn1-modules==0.2.8', 'pycparser==2.20', 'pydot==1.4.1', 'pyparsing==2.4.7', 'python-dotenv==0.14.0', 'pytz==2020.1', 'requests==2.24.0', 'requests-oauthlib==1.3.0', 'rsa==4.6', 'scikit-learn==0.23.2', 'scipy==1.4.1', 'six==1.15.0', 'tabulate==0.8.7', 'tensorboard==2.3.0', 'tensorboard-plugin-wit==1.7.0', 'tensorflow==2.3.1', 'tensorflow-cloud==0.1.6', 'tensorflow-datasets==3.0.0', 'tensorflow-estimator==2.3.0', 'tensorflow-hub==0.9.0', 'tensorflow-metadata==0.24.0', 'termcolor==1.1.0', 'terminaltables==3.1.0', 'threadpoolctl==2.1.0', 'tqdm==4.49.0', 'uritemplate==3.0.1', 'urllib3==1.25.10', 'websocket-client==0.57.0', 'Werkzeug==1.0.1', 'wrapt==1.12.1']

setup(
    name='bert-faqclass-setup',
    version='0.1',
    description='bert-faqclass application package',
    install_requires=requirements,
    packages=find_packages(),
    include_package_data=True,
)