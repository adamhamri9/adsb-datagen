import pytest

from src.adsb_generator.message import ADSBMessage, MessageType


class TestMessageType:
    def test_has_identification(self):
        assert MessageType.IDENTIFICATION.value == "identification"

    def test_has_surface_position(self):
        assert MessageType.SURFACE_POSITION.value == "surface_position"

    def test_has_airborne_position(self):
        assert MessageType.AIRBORNE_POSITION.value == "airborne_position"

    def test_has_airborne_velocity(self):
        assert MessageType.AIRBORNE_VELOCITY.value == "airborne_velocity"

    def test_all_types_are_unique(self):
        values = [t.value for t in MessageType]
        assert len(values) == len(set(values))


class TestADSBMessageProbabilities:
    def test_default_initialization(self):
        msg = ADSBMessage()

        assert msg.message_type_probs == {
            MessageType.IDENTIFICATION: 0.25,
            MessageType.SURFACE_POSITION: 0.25,
            MessageType.AIRBORNE_POSITION: 0.25,
            MessageType.AIRBORNE_VELOCITY: 0.25,
        }

    def test_custom_probabilities(self):
        custom = {
            MessageType.IDENTIFICATION: 0.1,
            MessageType.SURFACE_POSITION: 0.2,
            MessageType.AIRBORNE_POSITION: 0.3,
            MessageType.AIRBORNE_VELOCITY: 0.4,
        }
        msg = ADSBMessage(message_type_probs=custom)

        assert msg.message_type_probs == custom

    def test_default_probs_sum_to_one(self):
        msg = ADSBMessage()
        total = sum(msg.message_type_probs.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_valid_custom_probs_accepted(self):
        custom = {
            MessageType.IDENTIFICATION: 0.5,
            MessageType.SURFACE_POSITION: 0.2,
            MessageType.AIRBORNE_POSITION: 0.2,
            MessageType.AIRBORNE_VELOCITY: 0.1,
        }
        msg = ADSBMessage(message_type_probs=custom)
        assert sum(msg.message_type_probs.values()) == pytest.approx(1.0, abs=0.01)

    def test_rejects_probs_summing_too_high(self):
        bad = {
            MessageType.IDENTIFICATION: 0.5,
            MessageType.SURFACE_POSITION: 0.5,
            MessageType.AIRBORNE_POSITION: 0.5,
            MessageType.AIRBORNE_VELOCITY: 0.5,
        }
        with pytest.raises(ValueError, match="Sum of probabilities"):
            ADSBMessage(message_type_probs=bad)

    def test_rejects_probs_summing_too_low(self):
        bad = {
            MessageType.IDENTIFICATION: 0.01,
            MessageType.SURFACE_POSITION: 0.01,
            MessageType.AIRBORNE_POSITION: 0.01,
            MessageType.AIRBORNE_VELOCITY: 0.01,
        }
        with pytest.raises(ValueError, match="Sum of probabilities"):
            ADSBMessage(message_type_probs=bad)

    def test_validation_is_called_on_init(self):
        bad = {
            MessageType.IDENTIFICATION: 0.9,
            MessageType.SURFACE_POSITION: 0.9,
            MessageType.AIRBORNE_POSITION: 0.9,
            MessageType.AIRBORNE_VELOCITY: 0.9,
        }
        with pytest.raises(ValueError):
            ADSBMessage(message_type_probs=bad)


class TestBuildIdentificationMessage:
    def setup_method(self):
        self.msg = ADSBMessage()

    def test_returns_int(self):
        result = self.msg._build_identification_message()
        assert isinstance(result, int)

    def test_tc_in_valid_range(self):
        result = self.msg._build_identification_message()
        tc = (result >> 51) & 0x7
        assert 1 <= tc <= 4

    def test_ca_in_valid_range(self):
        result = self.msg._build_identification_message()
        ca = (result >> 48) & 0x7
        assert 0 <= ca <= 7

    def test_fits_in_56_bits(self):
        result = self.msg._build_identification_message()
        assert result < (1 << 56)

    def test_is_non_negative(self):
        result = self.msg._build_identification_message()
        assert result >= 0

    def test_multiple_calls_produce_varied_tc(self):
        tc_values = set()
        for _ in range(100):
            result = self.msg._build_identification_message()
            tc_values.add((result >> 51) & 0x7)
        assert len(tc_values) > 1

    def test_multiple_calls_produce_varied_ca(self):
        ca_values = set()
        for _ in range(100):
            result = self.msg._build_identification_message()
            ca_values.add((result >> 48) & 0x7)
        assert len(ca_values) > 1

    def test_callsign_encoding_occupies_lower_48_bits(self):
        result = self.msg._build_identification_message()
        callsign_bits = result & 0xFFFFFFFFFFFF
        assert callsign_bits < (1 << 48)


class TestBuildSurfacePositionMessage:
    def setup_method(self):
        self.msg = ADSBMessage()

    def test_returns_int(self):
        result = self.msg._build_surface_position_message()
        assert isinstance(result, int)

    def test_fits_in_56_bits(self):
        result = self.msg._build_surface_position_message()
        assert result < (1 << 56)

    def test_tc_in_valid_range(self):
        for _ in range(100):
            result = self.msg._build_surface_position_message()
            tc = (result >> 51) & 0x1F
            assert 5 <= tc <= 8

    def test_mov_in_valid_range(self):
        for _ in range(100):
            result = self.msg._build_surface_position_message()
            mov = (result >> 44) & 0x7F
            assert 0 <= mov <= 127

    def test_s_bit_is_boolean(self):
        for _ in range(50):
            result = self.msg._build_surface_position_message()
            s = (result >> 43) & 0x1
            assert s in (0, 1)

    def test_t_bit_is_boolean(self):
        for _ in range(50):
            result = self.msg._build_surface_position_message()
            t = (result >> 35) & 0x1
            assert t in (0, 1)

    def test_f_bit_is_boolean(self):
        for _ in range(50):
            result = self.msg._build_surface_position_message()
            f = (result >> 34) & 0x1
            assert f in (0, 1)

    def test_multiple_calls_produce_varied_tc(self):
        tc_values = set()
        for _ in range(100):
            result = self.msg._build_surface_position_message()
            tc_values.add((result >> 51) & 0x7)
        assert len(tc_values) > 1

    def test_multiple_calls_produce_varied_mov(self):
        mov_values = set()
        for _ in range(100):
            result = self.msg._build_surface_position_message()
            mov_values.add((result >> 44) & 0x7F)
        assert len(mov_values) > 1

    def test_multiple_calls_produce_varied_f(self):
        f_values = set()
        for _ in range(100):
            result = self.msg._build_surface_position_message()
            f_values.add((result >> 34) & 0x1)
        assert len(f_values) > 1


class TestBuildAirbornePositionMessage:
    def setup_method(self):
        self.msg = ADSBMessage()

    def test_returns_int(self):
        result = self.msg._build_airborne_position_message()
        assert isinstance(result, int)

    def test_fits_in_56_bits(self):
        result = self.msg._build_airborne_position_message()
        assert result < (1 << 56)

    def test_tc_in_valid_range(self):
        for _ in range(200):
            result = self.msg._build_airborne_position_message()
            tc = (result >> 51) & 0x1F
            assert tc in range(9, 19) or tc in range(20, 23)

    def test_ss_in_valid_range(self):
        for _ in range(100):
            result = self.msg._build_airborne_position_message()
            ss = (result >> 49) & 0x3
            assert 0 <= ss <= 3

    def test_saf_is_boolean(self):
        for _ in range(50):
            result = self.msg._build_airborne_position_message()
            saf = (result >> 48) & 0x1
            assert saf in (0, 1)

    def test_t_bit_is_boolean(self):
        for _ in range(50):
            result = self.msg._build_airborne_position_message()
            t = (result >> 35) & 0x1
            assert t in (0, 1)

    def test_f_bit_is_boolean(self):
        for _ in range(50):
            result = self.msg._build_airborne_position_message()
            f = (result >> 34) & 0x1
            assert f in (0, 1)

    def test_tc_excludes_19(self):
        tc_values = set()
        for _ in range(500):
            result = self.msg._build_airborne_position_message()
            tc_values.add((result >> 51) & 0x1F)
        assert 19 not in tc_values

    def test_multiple_calls_produce_varied_tc(self):
        tc_values = set()
        for _ in range(200):
            result = self.msg._build_airborne_position_message()
            tc_values.add((result >> 51) & 0x1F)
        assert len(tc_values) > 1

    def test_multiple_calls_produce_varied_f(self):
        f_values = set()
        for _ in range(100):
            result = self.msg._build_airborne_position_message()
            f_values.add((result >> 34) & 0x1)
        assert len(f_values) > 1


class TestBuildAirborneVelocityMessage:
    def setup_method(self):
        self.msg = ADSBMessage()

    def test_returns_int(self):
        result = self.msg._build_airborne_velocity_message()
        assert isinstance(result, int)

    def test_fits_in_56_bits(self):
        result = self.msg._build_airborne_velocity_message()
        assert result < (1 << 56)

    def test_tc_is_always_19(self):
        for _ in range(100):
            result = self.msg._build_airborne_velocity_message()
            tc = (result >> 51) & 0x1F
            assert tc == 19

    def test_st_in_valid_range(self):
        for _ in range(100):
            result = self.msg._build_airborne_velocity_message()
            st = (result >> 48) & 0x7
            assert 1 <= st <= 4

    def test_ic_is_boolean(self):
        for _ in range(50):
            result = self.msg._build_airborne_velocity_message()
            ic = (result >> 47) & 0x1
            assert ic in (0, 1)

    def test_ifr_is_boolean(self):
        for _ in range(50):
            result = self.msg._build_airborne_velocity_message()
            ifr = (result >> 46) & 0x1
            assert ifr in (0, 1)

    def test_nucv_in_valid_range(self):
        for _ in range(100):
            result = self.msg._build_airborne_velocity_message()
            nucv = (result >> 43) & 0x7
            assert 0 <= nucv <= 4

    def test_vrsrc_is_boolean(self):
        for _ in range(50):
            result = self.msg._build_airborne_velocity_message()
            vrsrc = (result >> 20) & 0x1
            assert vrsrc in (0, 1)

    def test_svr_is_boolean(self):
        for _ in range(50):
            result = self.msg._build_airborne_velocity_message()
            svr = (result >> 19) & 0x1
            assert svr in (0, 1)

    def test_res_is_always_zero(self):
        for _ in range(50):
            result = self.msg._build_airborne_velocity_message()
            res = (result >> 8) & 0x3
            assert res == 0

    def test_sdif_is_boolean(self):
        for _ in range(50):
            result = self.msg._build_airborne_velocity_message()
            sdif = (result >> 7) & 0x1
            assert sdif in (0, 1)

    def test_multiple_calls_produce_varied_st(self):
        st_values = set()
        for _ in range(200):
            result = self.msg._build_airborne_velocity_message()
            st_values.add((result >> 48) & 0x7)
        assert len(st_values) > 1

    def test_subtype_fits_22_bits(self):
        for _ in range(100):
            result = self.msg._build_airborne_velocity_message()
            subtype = (result >> 21) & 0x3FFFFF
            assert subtype < (1 << 22)

    def test_ground_speed_vew_dns_vns_fields_st1(self):
        for _ in range(100):
            result = self.msg._build_airborne_velocity_message()
            st = (result >> 48) & 0x7
            if st != 1:
                continue
            subtype = (result >> 21) & 0x3FFFFF
            dew = (subtype >> 21) & 0x1
            vew = (subtype >> 11) & 0x3FF
            dns = (subtype >> 10) & 0x1
            vns = subtype & 0x3FF
            assert dew in (0, 1)
            assert dns in (0, 1)
            assert 1 <= vew <= 1022
            assert 1 <= vns <= 1022

    def test_ground_speed_vew_dns_vns_fields_st2(self):
        for _ in range(100):
            result = self.msg._build_airborne_velocity_message()
            st = (result >> 48) & 0x7
            if st != 2:
                continue
            subtype = (result >> 21) & 0x3FFFFF
            dew = (subtype >> 21) & 0x1
            vew = (subtype >> 11) & 0x3FF
            dns = (subtype >> 10) & 0x1
            vns = subtype & 0x3FF
            assert dew in (0, 1)
            assert dns in (0, 1)
            assert 1 <= vew <= 1022
            assert 1 <= vns <= 1022

    def test_airspeed_hdg_t_ais_fields_st3(self):
        for _ in range(100):
            result = self.msg._build_airborne_velocity_message()
            st = (result >> 48) & 0x7
            if st != 3:
                continue
            subtype = (result >> 21) & 0x3FFFFF
            sh = (subtype >> 21) & 0x1
            hdg = (subtype >> 11) & 0x3FF
            t = (subtype >> 10) & 0x1
            ais = subtype & 0x3FF
            assert sh in (0, 1)
            assert t in (0, 1)
            assert 0 <= hdg <= 1023
            assert 1 <= ais <= 1022

    def test_airspeed_hdg_t_ais_fields_st4(self):
        for _ in range(100):
            result = self.msg._build_airborne_velocity_message()
            st = (result >> 48) & 0x7
            if st != 4:
                continue
            subtype = (result >> 21) & 0x3FFFFF
            sh = (subtype >> 21) & 0x1
            hdg = (subtype >> 11) & 0x3FF
            t = (subtype >> 10) & 0x1
            ais = subtype & 0x3FF
            assert sh in (0, 1)
            assert t in (0, 1)
            assert 0 <= hdg <= 1023
            assert 1 <= ais <= 1022


class TestBuild:
    IDENT_ONLY = {
        MessageType.IDENTIFICATION: 1.0,
        MessageType.SURFACE_POSITION: 0.0,
        MessageType.AIRBORNE_POSITION: 0.0,
        MessageType.AIRBORNE_VELOCITY: 0.0,
    }

    def setup_method(self):
        self.msg = ADSBMessage(message_type_probs=self.IDENT_ONLY)

    def test_returns_int(self):
        result = self.msg.build()
        assert isinstance(result, int)

    def test_result_is_112_bits(self):
        result = self.msg.build()
        assert 0 <= result < (1 << 112)

    def test_df_field_is_17(self):
        result = self.msg.build()
        df = (result >> 107) & 0x1F
        assert df == 17

    def test_ca_field_in_valid_range(self):
        result = self.msg.build()
        ca = (result >> 104) & 0x7
        assert 0 <= ca <= 7

    def test_icao_field_fits_24_bits(self):
        result = self.msg.build()
        icao = (result >> 80) & 0xFFFFFF
        assert 0 <= icao < (1 << 24)

    def test_returns_unique_values(self):
        results = {self.msg.build() for _ in range(20)}
        assert len(results) > 1

    def test_identification_tc_in_valid_range(self):
        for _ in range(50):
            result = self.msg.build()
            me = (result >> 24) & 0xFFFFFFFFFFFFFF
            tc = (me >> 51) & 0x7
            assert 1 <= tc <= 4

    def test_identification_ca_in_valid_range(self):
        for _ in range(50):
            result = self.msg.build()
            me = (result >> 24) & 0xFFFFFFFFFFFFFF
            ca = (me >> 48) & 0x7
            assert 0 <= ca <= 7

    def test_identification_callsign_fits_48_bits(self):
        for _ in range(20):
            result = self.msg.build()
            me = (result >> 24) & 0xFFFFFFFFFFFFFF
            callsign = me & 0xFFFFFFFFFFFF
            assert callsign < (1 << 48)

    def test_crc_is_valid(self):
        from src.adsb_generator.algorithms import ADSBAlgorithms
        for _ in range(20):
            result = self.msg.build()
            data_without_crc = result >> 24
            recalc = ADSBAlgorithms.calculate_crc(data_without_crc)
            assert recalc == result
