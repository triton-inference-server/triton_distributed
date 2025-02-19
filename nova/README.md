# NOVA

## Overview

NOVA is a library and tool designed to simplify the process of building and deploying multi-step and distributed inference pipelines. You can think of it as the frontend for [Triton Distributed](https://github.com/triton-inference-server/triton-distributed).

# Key Features
* **Python**: Write your pipelines in Python which allows for type safety and validation  
* **Local Development**: NOVA allows you to deploy your entire pipeline locally with single command
* **Graph Compilation**: NOVA ouputs an inference graph which can be compiled and versioned.
* **Containerization and Deployment**: NOVA leverages an open source containerization and deployment tool called [BentoML](https://github.com/bentoml/BentoML) for a developer friendly experience

# Getting Started

You can get started with NOVA by running the following commands:

```bash
mkdir nova-project
cd nova-project
uv init --no-readme  
```

This will create a new directory called `nova-project` and initialize a new uv project.

Next, go ahead and install nova

```bash
uv add nova 
```
This will install the nova package

# Hello World
Here we create a simple hello world pipeline and show some core nova building blocks.

## Step 1: Define your pipeline
In your main.py file, go ahead and add the following code:

```python 
from nova import nova_service, nova_api, depends, nova_endpoint
from pydantic import BaseModel

class Request(BaseModel):
    name: str

class Response(BaseModel):
    message: str


@nova_service
class Frontend:
    middle = depends(Middle)

    @nova_api(Request, Response)
    async def generate(self, text):
        async for response in self.middle.generate(text):
            yield f"Frontend: {response}"

@nova_service
class Middle:
    backend = depends(Backend)

    @nova_endpoint(Request, Response)
    async def generate(self, text):
        text = f"{text}-mid"
        async for response in self.backend.generate(text):
            yield f"Middle: {response}"

@nova_service
class Backend:
    @nova_endpoint(Request, Response)
    async def generate(self, text):
        text = f"{text}-backend"
        for token in text.split():
            yield f"Backend: {token}"
```

NOVA contains a set of core primatives that allow you to build this pipeline. Lets understand the different components here:

* `nova_service`: This is a decorator that tells NOVA that this is a service. A service is a deployable unit that can be run locally or remotely.
* `depends`: This is a decorator that tells NOVA to create an edge between another service. This allows you to use the target services endpoint as a function call.
* `nova_api`: This is a decorator that tells NOVA that this is an HTTP endpoint. This is primarily how you interact with your pipeline. 
* `nova_endpoint`: This is a decorator that tells NOVA that this is a distributed endpoint that registers on itself on a `DistributedRuntime`. More on that later

## Step 2: Compile your pipeline
Run the following command to compile your pipeline:

```bash
nova compile main:Frontend
```

This will perform type checks and create a new directory called `graph` with a file called `nova-graph.yaml` which contains a representation of your pipeline

## Step 3: Run your pipeline locally
Run the following command to run your pipeline locally:

```bash
nova serve main:Frontend
```

You can make requests to your pipeline by running the following command:

```bash
curl -X POST http://localhost:3000/generate -H "Content-Type: application/json" -d '{"name": "John"}'
```

## Step 4: Build your container