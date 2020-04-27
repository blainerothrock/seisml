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


def test_call():
    data = {'raw': [0, 0, 0]}
    tf = BaseTransform()
    data = tf(data)


def test_inplace():
    src = 'test_src'
    out = 'test_out'
    inplace = True

    data = {src: [0, 0, 0]}
    transformed = [1, 1, 1]

    tf = BaseTransform(source=src, output=out, inplace=inplace)
    data = tf.update(data, transformed)
    assert len(data.keys()) == 1
    assert data[src] == transformed
