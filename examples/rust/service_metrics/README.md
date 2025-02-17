# Service Metrics

This example extends the hello_world example by calling the `scrape_service` method
with the service name for the request response the client just issued a request.

The client can now observe some basic statistics about each instance of the service
begin hosted.

If you start two copies of the server, you will see two entries being emitted.
