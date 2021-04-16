# -*- coding: utf-8 -*-

import pytest
from p_detector.skeleton import fib

__author__ = "lee_jn"
__copyright__ = "lee_jn"
__license__ = "mit"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
