# Count

## Quickstart

To start count, simply point it at the namespace/component/endpoint trio that
you're interested in observing metrics from. This will scrape statistics from
the services associated with that endpoint, do some postprocessing on them,
and then publish an event with the postprocessed data.

```bash
# For more details, try TRD_LOG=debug
TRD_LOG=info cargo run -- --namespace triton-init --component backend --endpoint generate
```
