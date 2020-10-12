# Install the nightly version of tensorflow
FROM tensorflow/tensorflow:nightly

# Install requirements
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

# Copies the src code to the docker image.
COPY src /app/src/
COPY auth /app/auth/
COPY .env /app/
COPY config /app/config/

RUN mkdir /app/output
RUN mkdir /app/model
RUN mkdir /app/logs
RUN ls -lah /app/
RUN ls -lah /app/src/

ENV PYTHONPATH="/app/"

WORKDIR /app

# Set up the entry point to invoke the src.
ENTRYPOINT ["python", "./src/main.py"]