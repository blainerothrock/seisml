import pytest
from seisml.core.transforms import BaseTransform, TransformException


class TestBaseTransform:

    def test_initialization(self):
        src = 'test_src'
        out = 'test_out'
        inplace = True
        tf = BaseTransform(source=src, output=out, inplace=inplace)
        assert tf.source == src
        assert tf.output == out
        assert tf.inplace == inplace

    def test_call(self):
        data = {'raw': [0, 0, 0]}
        tf = BaseTransform()
        _ = tf(data)

    def test_inplace(self):
        src = 'test_src'
        out = 'test_out'
        inplace = True

        data = {src: [0, 0, 0]}
        transformed = [1, 1, 1]

        tf = BaseTransform(source=src, output=out, inplace=inplace)
        data = tf.update(data, transformed)
        assert len(data.keys()) == 1
        assert data[src] == transformed

    def test_params(self):
        with pytest.raises(TransformException) as _:
            tf = BaseTransform()
            tf(6)

        with pytest.raises(TransformException) as _:
            tf = BaseTransform(source='test')
            data = {'not_test': []}
            tf(data)