epochs:
- create arbitrary subclasses of service, dependency, and endpoint to see if we can string something together
- string in cli code into cai with differing entrypoint if nova service

```
uv init --package compoundai
uv add bentoml
uv run python # get a python shell
uv sync
uv run compoundai
```

testing that cai serve works via import instead
```
cd examples/basic_service
uv run cai serve --port 5005 service:PipelineService
```

- make changes to work with triton dist
    - add module to serve triton worker
    - modify getattr to match impl of current approach
    - modify serving to treat nova services differently

- let's test tomorrow

