#!/bin/bash

# Variables
IMAGE_NAME="bribrain-ppocr-new"            # Replace with your Docker image name
CONTAINER_NAME="ms-bribrain-ppocr-new"    # Replace with your Docker container name
DOCKERFILE_PATH="."                   # Path to the Dockerfile (default: current directory)
PORT_MAPPING="80:9000"                # Replace with your desired port mapping

# Build the Docker image
echo "Building Docker image: $IMAGE_NAME..."
docker build -t $IMAGE_NAME $DOCKERFILE_PATH
if [ $? -ne 0 ]; then
    echo "Error: Failed to build Docker image."
    exit 1
fi

# Stop and remove the old container if it exists
echo "Stopping and removing old container (if any)..."
if docker ps -q -f name=$CONTAINER_NAME > /dev/null; then
    docker stop $CONTAINER_NAME
    echo "Stopped container: $CONTAINER_NAME"
fi

if docker ps -a -q -f name=$CONTAINER_NAME > /dev/null; then
    docker rm $CONTAINER_NAME
    echo "Removed container: $CONTAINER_NAME"
fi

# Start a new container
echo "Starting a new container: $CONTAINER_NAME..."
docker run -d --name  $CONTAINER_NAME -p $PORT_MAPPING --restart always $IMAGE_NAME
if [ $? -ne 0 ]; then
    echo "Error: Failed to start new container."
    exit 1
fi

echo "Deployment successful. Container $CONTAINER_NAME is up and running."
