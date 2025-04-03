# Dockerfile for Docker version (>=20.10)

This Dockerfile allows you to build an image to based on the base layer `nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04` 
with torch and GPU support for the full **deep**doctection suite.

From the repo folder create an image

```
docker build -t deepdoctection/deepdoctection:<your-tag> -f docker/pytorch-gpu/Dockerfile .
``` 

Then start running a container. You can specify a volume for the cache, so that you do not need to download all weights
and configs once you re-start the container. 

```
docker run --name=dd-gpu --gpus all -v /host/to/dir:/home/files -v /path/to/cache:/root/.cache/deepdoctection -d -it deepdoctection/deepdoctection:<your-tag>
```

# Getting a Docker image from the Docker hub for the last published release

With the release of version v.0.27.0, we are starting to provide Docker images for the full installation. 
This is due to the fact that the requirements and dependencies are complex and even the construction of Docker images 
can lead to difficulties.

## Pulling images from the Docker hub:

```
docker pull deepdoctection/deepdoctection:<release_tag>
```

The container can be started with the above `docker run` command.

## Starting a container with docker compose

We provide a `docker-compose.yaml` file to start the generated image pulled from the hub. In order to use it, replace 
first the image argument with the tag, you want to use. Second, in the `.env` file, set the two environment variables:

`CACHE_HOST`: Model weights/configuration files, as well as potentially datasets, are not baked into the image during 
the build time, but mounted into the container with volumes. For a local installation, this is 
usually `~/.cache/deepdoctection`.

`WORK_DIR`: A temporary directory where documents can be loaded and also mounted into the container.

The container can be started as usual, for example, with:

```
docker compose up -d
```

Using the interpreter of the container, you can then run something like this:

```python
import deepdoctection as dd

if __name__=="__main__":

    analyzer = dd.get_dd_analyzer()

    df = analyzer.analyze(path = "/home/files/your_doc.pdf")
    df.reset_state()

    for dp in df:
        print(dp.text)
```



