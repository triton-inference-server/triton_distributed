from fastapi import FastAPI
from triton_api_server.open_ai.server import create_app


def create_openai_app(request_plane, data_plane):
    # Possibly define a "RemoteOperator" that points to your "encoder_decoder"
    # Or pass a specialized "BaseTriton3Connector" that uses request_plane/data_plane
    # Then call into your old code
    connector = MyTritonDistributedConnector(request_plane, data_plane)
    app = FastAPI()
    create_app(connector, app)
    return app
