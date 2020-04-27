import pytest
from seisml.core.transforms import BaseTransform


def test_initialization():
    src = 'test_src'
    out = 'test_out'
    inplace = True
    tf = BaseTransform(source=src, output=out, inplace=inplace)
    assert tf.source == src
    assert tf.output == out
    assert tf.inplace == inplace
