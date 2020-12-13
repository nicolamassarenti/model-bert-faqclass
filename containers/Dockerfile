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

ENV GOOGLE_APPLICATION_CREDENTIAL='{   "type": "service_account",   "project_id": "bert-faqclass",   "private_key_id": "a96dec925432659635bdda13fb8d10d7e5c27ff7",   "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDKw1eZFZiDFzCE\nBC4IaNMnG7MY1qhuWAwItj893rUiSxaCzinTXacztHymlaONgsG2/P8mpL7JcFU9\nvVzXqb6SGCMf6JCIwXUjegCf2STNXgTAZPD8gPVUq0kFD2YEty+NrN8Ds3RvPzdl\nU2hqQcpI5VjDPCUF5+CYMbktJL09ameNVIKmNRO6DtZhdN7X0BB9GfjSYYBT7c5O\nDPpWGfZuCJSC+e1f4j0XbsaICH15TubGMJ8POqxwwn4hWoUWVEuPMTL/J8Z3Mxdf\nfo9F/nM8NuHMc3isxgMfWr95vq8FGTeZHxwddIjmuQU7xETCdFhl0iU+droFdCZZ\netZ9j/xVAgMBAAECggEAWm93Dh3eOLXqmy88Et8iMqK9plnzpMebknUqzt8avhFs\nmGn4qL/wN/UL1ccPPCoyWXoNGYDIYZM7Sb7AfqZEi+e9Htm9QWm749diGOmiawDA\nsMk03QJtqdZ9yoZ8M6U6zf9pNr7fM1EoVYa2nxH59bppdSbJwpESEP+4foSQ7SdR\njM+5nC9XsPQzByXi28U2YXszaRzG6UW5Xfe2kGz9++FVAmk2BAY+U4xr0iWRlfzc\nNqGpskf9rIUtZvqzonCw5xfrWDCZavKjt8B5DKS6wZDbm3uyWM00XB0OSUpnDQVY\nmQOK0C89+rDnMkjD6iwuVa0wyj/U32lkah5KJvOebwKBgQDkc1r2IDT++BgH1M6O\nPszvn6Jqu0KqrQEfMqW3Gjm2uYJQprikhBftte5WkLeKdSc+UY/lYSsb7TVVjybx\nXN+qL+8Iz+QoPLPgskfW1kfKqtivujVae3/RtRQbnkvMxz9aKwsR4gkaNbOEFzOK\nF5hb0rDf6skfqrRQNjihs0b7wwKBgQDjNvhVDjslXVTfPTPTfjdxXMJTw4qcKg1Z\nnCmUdoBZjRNmnCGunYuJ4a+GmJmEDVwqKmezAt8TjPRS4AheW0Ge89EyOlWvUbxX\nwaYY7SWs6eEicsvyMmGyL2xPGm85bG1GFspq5XJOU7TQWAyoO4A2NBV04zhk1jVN\n6WWzLcTeBwKBgQDDfRb8gOj4ujJOGUO8I29ZKiBg9ueAh9Tq7nsyKWXzv+EMxO5C\nEassRj+yodn9yGyRflgFcFXkauL12S744YGf7667urJB4FgI0mZ5KBp9Ch+smEpp\nqD7jo/GWBKpWjLXgoe6yGduyKvv376ax56kAAlZK9m2vBHOSJCPn8zKt4wKBgHNi\nWe/Pn0eVQuFvEh+HzB38bg7dkUlGrarg+9iR6RIQ61ZDWXN0PAVUH6/pVKRTqIcd\nK7YjcidagESJ0pYgH43Z4VVxfEvg5RVnmLK101khGBxHUbx3y1cScB0nnm352vAp\n6LcqN8cBCSR4Yk0Ykj50aqUGMoA6q60aJnQ8fOoHAoGBAIgOaI1J5qjqvmcayz3N\nF4/AXPDiOqcQ3AvL+IMeB7TvJy6PbKrYEoKykbTXNXpZ/ASeXxJ6pI2tQMV8KykQ\nTbpsG7vvhFj2W+Em1V0RnbJI3CaxjANiIUioNtm7V8TIFi7MfMOeZIlA6U23xvvv\nSW2i/EqiYlJ06SbU32GGd9Gc\n-----END PRIVATE KEY-----\n",   "client_email": "be-dashboard-bert-faqclass@bert-faqclass.iam.gserviceaccount.com",   "client_id": "108074965435388117452",   "auth_uri": "https://accounts.google.com/o/oauth2/auth",   "token_uri": "https://oauth2.googleapis.com/token",   "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",   "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/be-dashboard-bert-faqclass%40bert-faqclass.iam.gserviceaccount.com" } '

# Run the application
ENTRYPOINT ["python", "-m", "bert_faqclass"]