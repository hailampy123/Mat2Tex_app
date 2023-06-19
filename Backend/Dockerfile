
FROM python:3.9-slim

# Install libraries
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt && \
    rm ./requirements.txt
# RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Setup container directories
RUN mkdir /app

# Copy local code to the container
COPY . ./app

# launch server with gunicorn
WORKDIR /app
EXPOSE 8080
CMD ["python", "app.py"]