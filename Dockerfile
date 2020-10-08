# Install the nightly version of tensorflow
FROM tensorflow/tensorflow:nightly
WORKDIR /root

# Install requirements
COPY requirements.txt /root/
RUN pip install -r requirements.txt

# Copies the src code to the docker image.
COPY . ./

# Set up the entry point to invoke the src.
ENTRYPOINT ["python", "trainer/task.py"]