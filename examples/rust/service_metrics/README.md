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
ServiceSet { services: [ServiceInfo { name: "triton_init_backend_720278f8", id: "j6n37goJog3df2PMkQK1Ry", version: "0.0.1", started: "2025-02-18T20:51:01.40830026Z", endpoints: [EndpointInfo { name: "triton_init_backend_720278f8-generate-694d94fc30dbb562", subject: "triton_init_backend_720278f8.generate-694d94fc30dbb562", data: Some(Metrics(Object {"average_processing_time": Number(67387), "last_error": String(""), "num_errors": Number(0), "num_requests": Number(1), "processing_time": Number(67387), "queue_group": String("q")})) }] }] }
```
