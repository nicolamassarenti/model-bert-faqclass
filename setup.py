from setuptools import find_packages
from setuptools import setup
import os


# Read requirements
requirements_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")
with open(requirements_file, "r") as f:
    requirements = f.read().splitlines()


setup(
    name='bert-faqclass-setup',
    version='0.1',
    description='Bert-faqclass application package',
    install_requires=requirements,
    packages=find_packages(),
    include_package_data=True,
)