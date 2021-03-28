FROM tensorflow/tensorflow:2.3.1

ARG PACKAGE_PATH=./package

# Setting working dir
WORKDIR application

# Pre-install pacakges before installing package to optimize caching
COPY ${PACKAGE_PATH}/requirements.txt .
RUN pip install -r requirements.txt

# Preparing code for package installation
COPY ${PACKAGE_PATH}/setup.py .
COPY ${PACKAGE_PATH}/MANIFEST.in .
COPY ${PACKAGE_PATH}/bert_faqclass bert_faqclass/

# Installing package
RUN pip install .

# Run the application
ENTRYPOINT ["python", "-m", "bert_faqclass"]