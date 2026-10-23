"""Small-block execution retains numerical recovery and profiling contracts."""
from unittest.mock import patch

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from two_stage_sgd_dwd import _prediction, prediction_profile


@pytest.mark.parametrize('n,p,q,batch,expected', [
    # Inclusive block-row limit, including a requested batch above that limit.
    (64,32,63,256,63), (64,32,64,256,64), (64,32,65,256,0),
    (64,32,256,256,0), (65,32,64,64,0),
    # Inclusive basis-element limit, with feature count held constant.
    (2047,1,64,64,64), (2048,1,64,64,64), (2049,1,64,64,0),
    # Chunking and tails decide execution by actual block length.
    (64,32,65,64,65), (64,32,129,64,129),
    (64,32,128,65,63), (64,32,129,65,64), (64,32,130,65,0),
    (64,32,130,66,64), (64,32,131,66,0),
    (128,16,257,256,1), (128,16,257,257,0),
    (1,33,8,256,0), (512,4,260,256,4),
    (513,4,260,256,0), (8,3,513,512,1),
])
def test_shape_boundaries_and_tail_dispatch(n,p,q,batch,expected):
    rng = np.random.default_rng(431)
    query, basis = rng.normal(size=(q,p)), rng.normal(size=(n,p))
    coef = np.abs(rng.normal(size=n))
    args = query,basis,coef,1.,1/p,batch
    before = [a.tobytes() for a in (query,basis,coef)]
    multiplied_rows = []
    original_multiply = np.matmul

    def observed_multiply(left, right, *values, **keywords):
        multiplied_rows.append(len(left))
        return original_multiply(left, right, *values, **keywords)

    with patch.object(_prediction.np, 'matmul', side_effect=observed_multiply):
        with prediction_profile() as records:
            actual = _prediction.rbf_prediction_scores_blas(*args)
    direct = _prediction.rbf_prediction_scores(*args)
    np.testing.assert_allclose(actual,direct,rtol=2e-12,atol=2e-13)
    event = records[0]['fallbacks'].get('execution_direct',{})
    assert event.get('rows_completed',0) == expected
    assert sum(multiplied_rows) == q - expected
    assert records[0]['rows_completed'] == q
    assert before == [a.tobytes() for a in (query,basis,coef)]


def test_direct_execution_retains_near_constant_recovery():
    args = (np.array([[.25],[.75]]),np.array([[0.],[1.]]),
            np.array([1.,-1.]),0.,1e-30,256)
    with prediction_profile() as records:
        actual = _prediction.rbf_prediction_scores_blas(*args)
    assert actual[0] > 0 and actual[1] < 0
    assert records[0]['fallbacks']['execution_direct']['rows_completed'] == 2
    assert records[0]['fallbacks']['near_constant_score']['rows_completed'] == 2


def test_direct_execution_reuses_buffer_and_does_not_hide_failure():
    query = np.arange(22.).reshape(11,2)
    basis = np.ones((3,2))
    args = (query,basis,np.ones(3),1.,.2,4)
    pointers = []
    def observed(left,right,**kwargs):
        pointers.append(kwargs['out'].ctypes.data)
        return cdist(left,right,**kwargs)
    with patch.object(_prediction,'cdist',side_effect=observed):
        _prediction.rbf_prediction_scores_blas(*args)
    assert len(pointers) == 3 and len(set(pointers)) == 1
    with patch.object(_prediction,'cdist',side_effect=FloatingPointError('probe')):
        with prediction_profile() as records:
            with pytest.raises(FloatingPointError,match='probe'):
                _prediction.rbf_prediction_scores_blas(*args)
    assert records[0]['rows_completed'] == 0
    assert records[0]['fallbacks']['execution_direct']['rows_completed'] == 0
