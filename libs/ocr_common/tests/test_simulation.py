from ocr_common.simulation import simulated_delay_seconds


def test_delay_token_is_read_only_when_enabled():
    assert simulated_delay_seconds("delay20s-npwp.jpg", enabled=True) == 20.0
    assert simulated_delay_seconds("DELAY5S.png", enabled=True) == 5.0
    assert simulated_delay_seconds("delay20s-npwp.jpg", enabled=False) == 0.0
    assert simulated_delay_seconds("npwp.jpg", enabled=True) == 0.0
    assert simulated_delay_seconds(None, enabled=True) == 0.0


def test_delay_is_capped():
    assert simulated_delay_seconds("delay9999s.jpg", enabled=True) == 120.0
