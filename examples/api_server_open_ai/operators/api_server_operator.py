import threading

import uvicorn
from triton_distributed.worker import Operator, RemoteInferenceRequest

# Suppose you have your older openai code in a local `server.py` or something
from .server import create_openai_app  # your old code that returns a FastAPI instance


class ApiServerOperator(Operator):
    def __init__(
        self,
        name,
        version,
        triton_core,
        request_plane,
        data_plane,
        parameters,
        repository,
        logger,
    ):
        self._triton_core = triton_core
        self._request_plane = request_plane
        self._data_plane = data_plane
        self._params = parameters
        self._logger = logger

        # Prepare or config the app
        self.app = create_openai_app(self._request_plane, self._data_plane)
        # ^ In your old code, you might pass a "connector" or a "RemoteOperator" reference

        # The simplest approach: spawn uvicorn in a background thread
        self.server_thread = None
        self.should_stop = False
        self.server_thread = threading.Thread(target=self.start_server)

    async def execute(self, requests: list[RemoteInferenceRequest]):
        """
        This can remain effectively no-op for typical requests. Or it can do
        something if you want to handle requests from the request plane.

        But mostly, the purpose is that once the worker is started, we
        spawn the server. The requests come in via HTTP, not the request plane.
        """
        self._logger.info(
            "API Server operator ignoring direct requests, it's purely for hosting HTTP endpoints."
        )
        for req in requests:
            await req.response_sender().send(final=True)  # or respond with NotSupported

    def start_server(self):
        """
        Launch uvicorn in a background thread or so
        """
        config = uvicorn.Config(self.app, host="0.0.0.0", port=8080, log_level="info")
        self.server = uvicorn.Server(config)
        self._logger.info("Starting uvicorn server for openai endpoints.")
        self.server.run()

    def __del__(self):
        # Attempt to gracefully stop uvicorn if still running
        self._logger.info("Stopping uvicorn server (API server operator cleanup)")
        if self.server:
            self.server.should_exit = True

    # Optionally override `start` if you want an immediate background thread
    # But typically the Worker is managing the operator lifecycle.
    # So you might rely on an event loop or some on-start method to do this.
