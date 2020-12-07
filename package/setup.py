from setuptools import find_packages
from setuptools import setup
import os

__this_location = os.path.abspath(os.path.realpath(__file__))
REQUIREMENTS_LOCATION = os.path.join(os.path.dirname(__this_location), "requirements.txt")

# Function to retrieve resources files
def _get_resources(package_name):
    # Get all the resources (also on nested levels)
    res_paths = os.path.join(package_name, "resources")
    all_resources = [os.path.join(folder, file) for folder, _, files in os.walk(res_paths) for file in files]
    # Remove the prefix: start just from "resources"
    return [resource[resource.index("resources"):] for resource in all_resources]


with open(REQUIREMENTS_LOCATION) as f:
    requirements = f.read().splitlines()

setup(
    name='bert_faqclass',
    version='0.1',
    description='Bert-faqclass application package',
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
    package_data={
        'bert_faqclass': _get_resources(package_name='bert_faqclass'),
        "tests": ["stubs/*"]
    }
)
