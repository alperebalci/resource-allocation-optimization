import numpy as np

from dea_or import DataEnvelopmentAnalysis


def example():
    # One input, one output. DMU 0 is pure-scale inefficient under CRS but
    # technically efficient under VRS. DMU 2 is dominated by DMU 1.
    inputs = np.array([[1.0], [2.0], [3.0]])
    outputs = np.array([[1.0], [3.0], [3.0]])
    return DataEnvelopmentAnalysis(inputs, outputs)


def test_ccr_identifies_scale_inefficiency():
    dea = example()
    a = dea.solve(0, returns_to_scale="crs")
    assert np.isclose(a.efficiency, 2.0 / 3.0, atol=1e-7)
    assert not a.efficient


def test_bcc_separates_pure_technical_from_scale_efficiency():
    dea = example()
    a = dea.solve(0, returns_to_scale="vrs")
    assert np.isclose(a.efficiency, 1.0, atol=1e-7)
    assert a.efficient


def test_dominated_dmu_is_inefficient_and_has_valid_peer_projection():
    dea = example()
    c = dea.solve(2, returns_to_scale="vrs")
    assert np.isclose(c.efficiency, 2.0 / 3.0, atol=1e-7)
    projected_x = c.lambdas @ dea.inputs
    projected_y = c.lambdas @ dea.outputs
    assert projected_x[0] <= c.efficiency * dea.inputs[2, 0] + 1e-8
    assert projected_y[0] >= dea.outputs[2, 0] - 1e-8
