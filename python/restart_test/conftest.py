from infinity_runner import InfinityRunner
from restart_timeout import *

import logging


def pytest_configure():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(process)d-%(thread)d %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def pytest_addoption(parser):
    parser.addoption(
        "--infinity_path",
        action="store",
        default="./build/Debug/src/infinity",
    )
    parser.addoption(
        "--config_path",
        action="store",
        default="./conf/infinity_conf.toml",
    )
    parser.addoption(
        "--builder_container",
        action="store",
    )


def pytest_generate_tests(metafunc):
    if "infinity_runner" in metafunc.fixturenames:
        infinity_path = metafunc.config.getoption("infinity_path")
        config_path = metafunc.config.getoption("config_path")

        runner = InfinityRunner(infinity_path, config_path)
        metafunc.parametrize("infinity_runner", [runner])


# def pytest_collection_modifyitems(config, items):
#     for item in items:
#         # Apply the decorator to each test function
#         test_name = item.name
#         item.obj = my_timeout(test_name)(item.obj)
