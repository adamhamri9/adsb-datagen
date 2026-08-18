import numpy as np
import pytest

from src.adsb_generator.generator import ADSBGenerator, ADSBSample
from src.adsb_generator.message import MessageType
from src.adsb_generator.channel import ChannelParams
from src.adsb_generator.encoder import TXParams


class TestADSBSample:
    def test_has_message_field(self):
        sample = ADSBSample(
            message=123,
            message_type=MessageType.AIRBORNE_POSITION,
            clean_signal=np.array([], dtype=np.complex64),
            tx_params={TXParams.AMPLITUDE: 1.0},
            channel_signal=np.array([], dtype=np.complex64),
            channel_params={ChannelParams.SNR_DB: 15.0},
        )
        assert sample.message == 123

    def test_has_message_type_field(self):
        sample = ADSBSample(
            message=0,
            message_type=MessageType.AIRBORNE_VELOCITY,
            clean_signal=np.array([], dtype=np.complex64),
            tx_params={},
            channel_signal=np.array([], dtype=np.complex64),
            channel_params={},
        )
        assert sample.message_type == MessageType.AIRBORNE_VELOCITY

    def test_has_clean_signal(self):
        sig = np.ones(10, dtype=np.complex64)
        sample = ADSBSample(
            message=0,
            message_type=MessageType.IDENTIFICATION,
            clean_signal=sig,
            tx_params={},
            channel_signal=np.array([], dtype=np.complex64),
            channel_params={},
        )
        np.testing.assert_array_equal(sample.clean_signal, sig)

    def test_has_channel_signal(self):
        sig = np.ones(10, dtype=np.complex64)
        sample = ADSBSample(
            message=0,
            message_type=MessageType.IDENTIFICATION,
            clean_signal=np.array([], dtype=np.complex64),
            tx_params={},
            channel_signal=sig,
            channel_params={},
        )
        np.testing.assert_array_equal(sample.channel_signal, sig)

    def test_has_tx_params(self):
        tx = {TXParams.AMPLITUDE: 0.5}
        sample = ADSBSample(
            message=0,
            message_type=MessageType.IDENTIFICATION,
            clean_signal=np.array([], dtype=np.complex64),
            tx_params=tx,
            channel_signal=np.array([], dtype=np.complex64),
            channel_params={},
        )
        assert sample.tx_params == tx

    def test_has_channel_params(self):
        ch = {ChannelParams.SNR_DB: 20.0}
        sample = ADSBSample(
            message=0,
            message_type=MessageType.IDENTIFICATION,
            clean_signal=np.array([], dtype=np.complex64),
            tx_params={},
            channel_signal=np.array([], dtype=np.complex64),
            channel_params=ch,
        )
        assert sample.channel_params == ch


class TestADSBGeneratorInit:
    def test_creates_builder(self):
        gen = ADSBGenerator(seed=42)
        assert gen.builder is not None

    def test_creates_encoder(self):
        gen = ADSBGenerator(seed=42)
        assert gen.encoder is not None

    def test_creates_channel(self):
        gen = ADSBGenerator(seed=42)
        assert gen.channel is not None

    def test_seed_from_parameter(self):
        gen = ADSBGenerator(seed=42)
        assert gen.seed == 42

    def test_seed_property(self):
        gen = ADSBGenerator(seed=99)
        assert gen.seed == 99

    def test_default_seed_is_random(self):
        gen1 = ADSBGenerator()
        gen2 = ADSBGenerator()
        assert gen1.seed != gen2.seed

    def test_subcomponents_share_seed(self):
        gen = ADSBGenerator(seed=42)
        assert gen.builder._seed == gen.encoder._seed == gen.channel._seed

    def test_custom_sample_rate_passed_to_encoder(self):
        gen = ADSBGenerator(sample_rate=4e6, seed=42)
        assert gen.encoder.sample_rate == 4e6

    def test_custom_sample_rate_passed_to_channel(self):
        gen = ADSBGenerator(sample_rate=4e6, seed=42)
        assert gen.channel.sample_rate == 4e6


class TestADSBGeneratorIter:
    def setup_method(self):
        self.gen = ADSBGenerator(seed=42)

    def test_is_iterable(self):
        assert hasattr(self.gen, '__iter__')

    def test_next_returns_adsbsample(self):
        gen = ADSBGenerator(seed=42)
        sample = next(iter(gen))
        assert isinstance(sample, ADSBSample)

    def test_message_is_int(self):
        sample = next(iter(self.gen))
        assert isinstance(sample.message, int)

    def test_message_type_is_valid(self):
        sample = next(iter(self.gen))
        assert sample.message_type in MessageType

    def test_clean_signal_is_ndarray(self):
        sample = next(iter(self.gen))
        assert isinstance(sample.clean_signal, np.ndarray)

    def test_channel_signal_is_ndarray(self):
        sample = next(iter(self.gen))
        assert isinstance(sample.channel_signal, np.ndarray)

    def test_tx_params_is_dict(self):
        sample = next(iter(self.gen))
        assert isinstance(sample.tx_params, dict)

    def test_channel_params_is_dict(self):
        sample = next(iter(self.gen))
        assert isinstance(sample.channel_params, dict)

    def test_clean_signal_is_complex(self):
        sample = next(iter(self.gen))
        assert np.iscomplexobj(sample.clean_signal)

    def test_channel_signal_is_complex(self):
        sample = next(iter(self.gen))
        assert np.iscomplexobj(sample.channel_signal)

    def test_clean_and_channel_same_length(self):
        sample = next(iter(self.gen))
        assert sample.clean_signal.shape == sample.channel_signal.shape

    def test_clean_signal_non_empty(self):
        sample = next(iter(self.gen))
        assert len(sample.clean_signal) > 0

    def test_channel_params_has_all_keys(self):
        sample = next(iter(self.gen))
        assert len(sample.channel_params) == len(ChannelParams)

    def test_tx_params_has_amplitude(self):
        sample = next(iter(self.gen))
        assert TXParams.AMPLITUDE in sample.tx_params

    def test_multiple_yields_distinct_messages(self):
        gen = ADSBGenerator(seed=42)
        it = iter(gen)
        messages = {next(it).message for _ in range(10)}
        assert len(messages) > 1

    def test_multiple_yields_varied_message_types(self):
        gen = ADSBGenerator(seed=42)
        it = iter(gen)
        types = {next(it).message_type for _ in range(100)}
        assert len(types) > 1

    def test_seeded_reproducibility(self):
        gen1 = ADSBGenerator(seed=99)
        gen2 = ADSBGenerator(seed=99)
        s1 = next(iter(gen1))
        s2 = next(iter(gen2))
        assert s1.message == s2.message
        assert s1.message_type == s2.message_type
        np.testing.assert_array_equal(s1.clean_signal, s2.clean_signal)
        np.testing.assert_array_equal(s1.channel_signal, s2.channel_signal)

    def test_different_seeds_different_output(self):
        gen1 = ADSBGenerator(seed=1)
        gen2 = ADSBGenerator(seed=2)
        s1 = next(iter(gen1))
        s2 = next(iter(gen2))
        assert s1.message != s2.message

    def test_yields_indefinitely(self):
        gen = ADSBGenerator(seed=42)
        it = iter(gen)
        for _ in range(100):
            next(it)
