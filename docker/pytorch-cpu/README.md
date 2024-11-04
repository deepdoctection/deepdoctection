# Dockerfile for Docker version (>=20.10)

This Dockerfile allows you to build an image to based on the base layer `python:3.9-slim` with torch for CPU and the 
full **deep**doctection suite installed for demonstration purposes.

From repo folder create an image

```
docker build -t dd:<your-tag> -f docker/pytorch-cpu/Dockerfile .
``` 

Then start running a container. Specify a host directory if you want to have some files mounted into the container

```
docker run -d -t --name=dd-deepdoctection-cpu -v /host/to/dir:/home/files dd:<your-tag>
```
