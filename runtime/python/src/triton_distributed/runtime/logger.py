# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import logging.config
from typing import Any

_LOGGER_NAME = "Triton Distributed Runtime"

_FHANDLER_CONFIG_TEMPLATE = {
    "class": "logging.FileHandler",
    "formatter": "standard",
}

_LOGGER_CONFIG_TEMPLATE = {"handlers": ["console"], "propagate": True}

_LOGGING_CONFIG_TEMPLATE = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            "datefmt": "%H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        }
    },
}


def get_logger_config(log_level=1, logger_name=_LOGGER_NAME, log_file=None):
    config_dict: dict[str, Any] = _LOGGING_CONFIG_TEMPLATE
    front = "%(asctime)s.%(msecs)03d %(filename)s:%(lineno)s"
    config_dict["formatters"]["standard"][
        "format"
    ] = f"{front} [{logger_name}] %(levelname)s: %(message)s"

    if log_file:
        fh_config_dict = _FHANDLER_CONFIG_TEMPLATE
        fh_config_dict["filename"] = str(log_file)
        config_dict["handlers"]["file"] = fh_config_dict

    logger_config: dict[str, Any] = _LOGGER_CONFIG_TEMPLATE
    if log_file:
        logger_config["handlers"].append("file")

    config_dict["loggers"] = {}
    config_dict["loggers"][logger_name] = logger_config

    return config_dict


# TODO: Add support for taking logging level as input as well.
def get_logger(log_level=1, logger_name=_LOGGER_NAME, log_file=None):
    if log_level == 0:
        level = logging.ERROR
    elif log_level == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG
    config_dict = get_logger_config(log_level, logger_name, log_file)
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(logger_name)
    logger.setLevel(level=level)
    return logger
