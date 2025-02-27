# Count

## Quickstart

To start `count`, simply point it at the namespace/component/endpoint trio that
you're interested in observing metrics from. This will scrape statistics from
the services associated with that endpoint, do some postprocessing on them,
and then publish an event with the postprocessed data.

```bash
# For more details, try TRD_LOG=debug
TRD_LOG=info cargo run -- --namespace triton-init --component backend --endpoint generate

# 2025-02-26T18:45:05.467026Z  INFO count: Creating unique instance of Count at triton-init/components/count/instance
# 2025-02-26T18:45:05.472146Z  INFO count: Scraping service triton_init_backend_720278f8 and filtering on subject triton_init_backend_720278f8.generate
# ...
```

With no matching endpoints running, you should see warnings in the logs:
```bash
2025-02-26T18:45:06.474161Z  WARN count: No endpoints found matching subject triton_init_backend_720278f8.generate
```

But after starting a matching endpoint, such as the
[service_metrics example](examples/rust/service_metrics/src/bin/server.rs),
you should see these warnings go away since the endpoint will automatically
get discovered.

Whether there are matching endpoints found or not, `count` will publish events, for example:
```
2025-02-26T18:45:46.501874Z  INFO count: Publishing event l2c.backend.generate on Namespace { name: "triton-init" } with ProcessedEndpoints { capacity_with_ids: [], load_avg: NaN, load_std: NaN, address: "backend.generate" }
```

However, the events may not be very useful until there are corresponding stats found from endpoints for processing.
