def pytest_collection_modifyitems(config, items):
    """
    Place the slowest tests at tne end of the test execution order if `order-tests` flag is set.
    Adapted from https://stackoverflow.com/a/54254338/8505509
    """
    if config.option.order_tests:
        items.sort(key=lambda item: 2 if item.get_closest_marker("slow") else 1)


def pytest_addoption(parser):
    parser.addoption(
        "--order-tests",
        action="store_true",
        help="Flag to place slow test functions at the end of the test execution order",
    )
