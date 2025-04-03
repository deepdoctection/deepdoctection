# Dockerfile for Docker version (>=20.10)

This Dockerfile allows you to build an image to based on the base layer `python:3.8-slim` with torch for CPU and the 
full **deep**doctection suite installed for demonstration purposes. A Jupyter notebook for can be used to
run sample code in the container.

From repo folder create an image

```
docker build -t dd:<your-tag> -f docker/pytorch-cpu-jupyter/Dockerfile .
``` 

Then start running a container. Specify a host directory if you want to have some files mounted into the container

```
docker run -d -t --name=dd-jupyter -v /host/to/dir:/home/files -p 8888:8888 dd:<your_tag> 
```

You can then access jupyter through `http://localhost:8888/tree`. You will have to enter a token you can access through
container logs:

```
docker logs <container id>
```
