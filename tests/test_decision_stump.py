import pytest

from src.decision_stump import DecisionStump


@pytest.fixture
def stump_1():
    return DecisionStump(0.5, 1)


@pytest.fixture
def stump_2():
    return DecisionStump(0.8, -1)


def test_value(stump_1, stump_2):
    stump_3 = DecisionStump(0.8, -1)

    assert stump_1 != stump_2
    assert stump_2 == stump_3
    assert hash(stump_1) != hash(stump_2)
    assert hash(stump_2) == hash(stump_3)


def test_call(stump_1, stump_2):
    assert stump_1(0.2) == -1.0
    assert stump_1(0.7) == 1.0

    assert stump_2(0.4) == 1.0
    assert stump_2(0.9) == -1.0




