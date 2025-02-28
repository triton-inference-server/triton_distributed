# Count

## Quickstart

To start `count`, simply point it at the namespace/component/endpoint trio that
you're interested in observing metrics from. This will scrape statistics from
the services associated with that endpoint, do some postprocessing on them,
and then publish an event with the postprocessed data.

```bash
# For more details, try TRD_LOG=debug
TRD_LOG=info cargo run --bin count -- --namespace triton-init --component backend --endpoint generate

# 2025-02-26T18:45:05.467026Z  INFO count: Creating unique instance of Count at triton-init/components/count/instance
# 2025-02-26T18:45:05.472146Z  INFO count: Scraping service triton_init_backend_720278f8 and filtering on subject triton_init_backend_720278f8.generate
# ...
```

With no matching endpoints running, you should see warnings in the logs:
```bash
2025-02-26T18:45:06.474161Z  WARN count: No endpoints found matching subject triton_init_backend_720278f8.generate
```

To see metrics published to a matching endpoint, you can use the
[mock_worker example](src/bin/mock_worker.rs) in this directory to launch
1 or more workers:
```bash
cargo run --bin mock_worker
```

After a matching endpoint gets started, you should see the warnings go away
since the endpoint will automatically get discovered.

Whether there are matching endpoints found or not, `count` will publish events, for example:
```
2025-02-26T18:45:46.501874Z  INFO count: Publishing event l2c.backend.generate on Namespace { name: "triton-init" } with ProcessedEndpoints { capacity_with_ids: [], load_avg: NaN, load_std: NaN, address: "backend.generate" }
```

However, the events may not be very useful unless there are corresponding
stats found from endpoints for processing.
