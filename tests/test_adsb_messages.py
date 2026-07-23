import pytest

from src.adsb_generator.messages import ADSBMessage, ADSBMessageType


class TestADSBMessageType:
    def test_has_identification(self):
        assert ADSBMessageType.IDENTIFICATION.value == "identification"

    def test_has_surface_position(self):
        assert ADSBMessageType.SURFACE_POSITION.value == "surface_position"

    def test_has_airborne_position(self):
        assert ADSBMessageType.AIRBORNE_POSITION.value == "airborne_position"

    def test_has_airborne_velocity(self):
        assert ADSBMessageType.AIRBORNE_VELOCITY.value == "airborne_velocity"

    def test_all_types_are_unique(self):
        values = [t.value for t in ADSBMessageType]
        assert len(values) == len(set(values))


class TestADSBMessageProbabilities:
    def test_default_initialization(self):
        msg = ADSBMessage()

        assert msg.message_type_probs == {
            ADSBMessageType.IDENTIFICATION: 0.25,
            ADSBMessageType.SURFACE_POSITION: 0.25,
            ADSBMessageType.AIRBORNE_POSITION: 0.25,
            ADSBMessageType.AIRBORNE_VELOCITY: 0.25,
        }

    def test_custom_probabilities(self):
        custom = {
            ADSBMessageType.IDENTIFICATION: 0.1,
            ADSBMessageType.SURFACE_POSITION: 0.2,
            ADSBMessageType.AIRBORNE_POSITION: 0.3,
            ADSBMessageType.AIRBORNE_VELOCITY: 0.4,
        }
        msg = ADSBMessage(message_type_probs=custom)

        assert msg.message_type_probs == custom

    def test_default_probs_sum_to_one(self):
        msg = ADSBMessage()
        total = sum(msg.message_type_probs.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_valid_custom_probs_accepted(self):
        custom = {
            ADSBMessageType.IDENTIFICATION: 0.5,
            ADSBMessageType.SURFACE_POSITION: 0.2,
            ADSBMessageType.AIRBORNE_POSITION: 0.2,
            ADSBMessageType.AIRBORNE_VELOCITY: 0.1,
        }
        msg = ADSBMessage(message_type_probs=custom)
        assert sum(msg.message_type_probs.values()) == pytest.approx(1.0, abs=0.01)

    def test_rejects_probs_summing_too_high(self):
        bad = {
            ADSBMessageType.IDENTIFICATION: 0.5,
            ADSBMessageType.SURFACE_POSITION: 0.5,
            ADSBMessageType.AIRBORNE_POSITION: 0.5,
            ADSBMessageType.AIRBORNE_VELOCITY: 0.5,
        }
        with pytest.raises(ValueError, match="Sum of probabilities"):
            ADSBMessage(message_type_probs=bad)

    def test_rejects_probs_summing_too_low(self):
        bad = {
            ADSBMessageType.IDENTIFICATION: 0.01,
            ADSBMessageType.SURFACE_POSITION: 0.01,
            ADSBMessageType.AIRBORNE_POSITION: 0.01,
            ADSBMessageType.AIRBORNE_VELOCITY: 0.01,
        }
        with pytest.raises(ValueError, match="Sum of probabilities"):
            ADSBMessage(message_type_probs=bad)

    def test_validation_is_called_on_init(self):
        bad = {
            ADSBMessageType.IDENTIFICATION: 0.9,
            ADSBMessageType.SURFACE_POSITION: 0.9,
            ADSBMessageType.AIRBORNE_POSITION: 0.9,
            ADSBMessageType.AIRBORNE_VELOCITY: 0.9,
        }
        with pytest.raises(ValueError):
            ADSBMessage(message_type_probs=bad)
