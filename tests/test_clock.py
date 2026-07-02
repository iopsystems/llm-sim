from vllm_sim.clock import VirtualClock


def test_starts_at_zero_by_default():
    clock = VirtualClock()
    assert clock.now() == 0.0


def test_starts_at_given_time():
    clock = VirtualClock(start=5.0)
    assert clock.now() == 5.0


def test_advance_increments_time():
    clock = VirtualClock()
    clock.advance(2.5)
    assert clock.now() == 2.5
    clock.advance(0.5)
    assert clock.now() == 3.0


def test_advance_rejects_negative():
    clock = VirtualClock()
    import pytest
    with pytest.raises(ValueError):
        clock.advance(-1.0)


def test_fast_forward_to_jumps_forward():
    clock = VirtualClock()
    clock.advance(1.0)
    clock.fast_forward_to(10.0)
    assert clock.now() == 10.0


def test_fast_forward_never_moves_backward():
    clock = VirtualClock()
    clock.advance(5.0)
    clock.fast_forward_to(3.0)  # earlier target is a no-op, time is monotonic
    assert clock.now() == 5.0
