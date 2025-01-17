def main(args):
    # define all your worker configs as before: encoder, decoder, etc.
    api_server_op = OperatorConfig(
        name="api_server",
        repository=str(example_dir.joinpath("operators", "api_server")),
        implementation="api_server_operator:ApiServerOperator",  # matches the .py file's operator class
        max_inflight_requests=1,
        parameters={},
    )

    api_server = WorkerConfig(operators=[api_server_op], name="api_server")

    # then your Deployment
    deployment = Deployment(
        [
            (api_server, 1),
        ],
        initialize_request_plane=True,
        log_dir=args.log_dir,
        log_level=args.log_level,
    )
    ...
    deployment.start()
    ...
