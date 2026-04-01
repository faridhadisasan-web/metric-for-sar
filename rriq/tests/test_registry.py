from rriq.filters.registry import get_filter

def test_get_filter_loads_median():
    # If __init__.py works, this should return without error
    func = get_filter("median")
    assert callable(func)

def test_get_filter_loads_lee():
    func = get_filter("lee")
    assert callable(func)
