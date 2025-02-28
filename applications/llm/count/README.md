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
1 or more workers that publish LLM Metrics:
```bash
# Can run multiple workers in separate shells
cargo run --bin mock_worker
```

After a matching endpoint gets started, you should see the warnings go away
since the endpoint will automatically get discovered.

When stats are found from the target endpoints being listened on, count will
aggregate and publish some metrics as both an event and to a prometheus web server:
```
2025-02-28T04:05:58.077901Z  INFO count: Aggregated metrics: ProcessedEndpoints { endpoints: [Endpoint { name: "worker-7587884888253033398", subject: "triton_init_backend_720278f8.generate-694d951a80e06bb6", data: ForwardPassMetrics { request_active_slots: 58, request_total_slots: 100, kv_active_blocks: 77, kv_total_blocks: 100 } }, Endpoint { name: "worker-7587884888253033401", subject: "triton_init_backend_720278f8.generate-694d951a80e06bb9", data: ForwardPassMetrics { request_active_slots: 71, request_total_slots: 100, kv_active_blocks: 29, kv_total_blocks: 100 } }], worker_ids: [7587884888253033398, 7587884888253033401], load_avg: 53.0, load_std: 24.0 }
```

To see the metrics being published in prometheus format, you can run:
```bash
curl localhost:9091/metrics

# # HELP llm_kv_blocks_active Active KV cache blocks
# # TYPE llm_kv_blocks_active gauge
# llm_kv_blocks_active{component="backend",endpoint="generate",worker_id="7587884888253033398"} 40
# llm_kv_blocks_active{component="backend",endpoint="generate",worker_id="7587884888253033401"} 2
# # HELP llm_kv_blocks_total Total KV cache blocks
# # TYPE llm_kv_blocks_total gauge
# llm_kv_blocks_total{component="backend",endpoint="generate",worker_id="7587884888253033398"} 100
# llm_kv_blocks_total{component="backend",endpoint="generate",worker_id="7587884888253033401"} 100
```
