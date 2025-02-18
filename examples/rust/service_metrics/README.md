# Service Metrics

This example extends the hello_world example by calling the `scrape_service` method
with the service name for the request response the client just issued a request.

The client can now observe some basic statistics about each instance of the service
begin hosted.

If you start two copies of the server, you will see two entries being emitted.

## Example Output
```
Annotated { data: Some("h"), id: None, event: None, comment: None }
Annotated { data: Some("e"), id: None, event: None, comment: None }
Annotated { data: Some("l"), id: None, event: None, comment: None }
Annotated { data: Some("l"), id: None, event: None, comment: None }
Annotated { data: Some("o"), id: None, event: None, comment: None }
Annotated { data: Some(" "), id: None, event: None, comment: None }
Annotated { data: Some("w"), id: None, event: None, comment: None }
Annotated { data: Some("o"), id: None, event: None, comment: None }
Annotated { data: Some("r"), id: None, event: None, comment: None }
Annotated { data: Some("l"), id: None, event: None, comment: None }
Annotated { data: Some("d"), id: None, event: None, comment: None }
Message { subject: Subject { bytes: b"_INBOX.lsBooCn9buJnXeyivLOPHh" }, reply: None, payload: b"{\"type\":\"io.nats.micro.v1.stats_response\",\"name\":\"triton_init_backend_720278f8\",\"id\":\"MmsvZvEPOaloyRrkoLpAnA\",\"version\":\"0.0.1\",\"started\":\"2025-02-17T22:15:48.35672076Z\",\"endpoints\":[{\"name\":\"triton_init_backend_720278f8-generate-694d94fc30dbb55a\",\"subject\":\"triton_init_backend_720278f8.generate-694d94fc30dbb55a\",\"num_requests\":1,\"num_errors\":0,\"processing_time\":52559,\"average_processing_time\":52559,\"last_error\":\"\",\"queue_group\":\"q\"}]}", headers: None, status: None, description: None, length: 468 }
```
