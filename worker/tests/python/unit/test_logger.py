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

import pytest

from triton_distributed.worker.log_formatter import LOGGER_NAME, setup_logger

logger = logging.getLogger(LOGGER_NAME)

MSG = "This is a sample message"

"""
Tests for Logging module
"""


def logging_function(logger):
    logger.info(MSG)
    logger.warning(MSG)
    try:
        raise Exception("This is an exception")
    except Exception:
        logger.exception(MSG)

    logger.error(MSG)
    logger.debug(MSG)


@pytest.fixture
def reset_logger(caplog):
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    loggers.append(logging.getLogger())
    for logger in loggers:
        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
        logger.setLevel(logging.NOTSET)
        logger.propagate = True
        caplog.clear()


@pytest.mark.parametrize(
    "log_level, expected_record_counts",
    [
        # For log-level 0 only error and exception should be recorded
        (0, 2),
        # For log-level 1 only info, error, exception and warning should be recorded
        (1, 4),
        # All logs(error, exception, info, debug and warning) should be printed for log-level 2
        (2, 5),
    ],
)
def test_logging(reset_logger, caplog, log_level, expected_record_counts):
    caplog.set_level(log_level)
    setup_logger(log_level=log_level)
    logging_function(logger)
    assert len(caplog.records) == expected_record_counts
