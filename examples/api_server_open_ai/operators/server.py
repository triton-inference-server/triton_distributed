from fastapi import FastAPI
from triton_api_server.open_ai.server import create_app

from .connector import MyTritonDistributedConnector


def create_openai_app(request_plane, data_plane):
    connector = MyTritonDistributedConnector(request_plane, data_plane)
    app = FastAPI()
    create_app(connector, app)
    return app
