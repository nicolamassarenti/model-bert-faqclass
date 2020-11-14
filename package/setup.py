from setuptools import find_packages
from setuptools import setup


setup(
    name='bert-faqclass-setup',
    version='0.1',
    description='Bert-faqclass application package',
    packages=find_packages(),
    install_requires=[
        "python-dotenv>=0.14.0"
    ],
    include_package_data=True,
    package_data={'model-bert-faqclass': ['.env']}
)
