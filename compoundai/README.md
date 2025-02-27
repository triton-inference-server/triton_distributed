# CompoundAI

Project components are nested under `deploy/compoundai` to conform to the eventual destination of cai components in the triton-distributed repository.

```
deploy/compoundai
├── api-server/         # Go-based API server
│   └── {cmd,internal,pkg,Dockerfile}
├── operator/         # Go-based controller
│   └── {cmd,internal,pkg,Dockerfile}
├── sdk/
│   └── python/         # Python SDK
├── examples/           # Usage examples
├── docs/
```