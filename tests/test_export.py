import json
import csv
import numpy as np
import pytest
from pathlib import Path

from src.adsb_generator.generator import ADSBGenerator, ADSBSample
from src.adsb_generator.types import MessageType, TXParams, ChannelParams


class TestSampleToDict:
    def setup_method(self):
        self.gen = ADSBGenerator(seed=42)

    def test_returns_dict(self):
        sample = next(iter(self.gen))
        result = self.gen._sample_to_dict(sample)
        assert isinstance(result, dict)

    def test_has_message_key(self):
        sample = next(iter(self.gen))
        result = self.gen._sample_to_dict(sample)
        assert "message" in result
        assert result["message"] == sample.message

    def test_has_message_type_key(self):
        sample = next(iter(self.gen))
        result = self.gen._sample_to_dict(sample)
        assert "message_type" in result
        assert result["message_type"] == sample.message_type.value

    def test_has_clean_signal_key(self):
        sample = next(iter(self.gen))
        result = self.gen._sample_to_dict(sample)
        assert "clean_signal" in result
        np.testing.assert_array_equal(result["clean_signal"], sample.clean_signal)

    def test_has_channel_signal_key(self):
        sample = next(iter(self.gen))
        result = self.gen._sample_to_dict(sample)
        assert "channel_signal" in result
        np.testing.assert_array_equal(result["channel_signal"], sample.channel_signal)

    def test_has_tx_params_key(self):
        sample = next(iter(self.gen))
        result = self.gen._sample_to_dict(sample)
        assert "tx_params" in result

    def test_has_channel_params_key(self):
        sample = next(iter(self.gen))
        result = self.gen._sample_to_dict(sample)
        assert "channel_params" in result

    def test_tx_params_keys_are_strings(self):
        sample = next(iter(self.gen))
        result = self.gen._sample_to_dict(sample)
        for key in result["tx_params"]:
            assert isinstance(key, str)

    def test_channel_params_keys_are_strings(self):
        sample = next(iter(self.gen))
        result = self.gen._sample_to_dict(sample)
        for key in result["channel_params"]:
            assert isinstance(key, str)

    def test_tx_params_enum_values_preserved(self):
        sample = next(iter(self.gen))
        result = self.gen._sample_to_dict(sample)
        for enum_key, val in sample.tx_params.items():
            assert enum_key.value in result["tx_params"]
            assert result["tx_params"][enum_key.value] == val

    def test_channel_params_enum_values_preserved(self):
        sample = next(iter(self.gen))
        result = self.gen._sample_to_dict(sample)
        for enum_key, val in sample.channel_params.items():
            assert enum_key.value in result["channel_params"]
            assert result["channel_params"][enum_key.value] == val

    def test_dict_has_six_keys(self):
        sample = next(iter(self.gen))
        result = self.gen._sample_to_dict(sample)
        assert len(result) == 6


class TestExportValidation:
    def test_raises_valueerror_with_empty_buffer(self):
        gen = ADSBGenerator(seed=42)
        with pytest.raises(ValueError, match="No data to export"):
            gen.export("/tmp/test.npz")

    def test_raises_valueerror_with_none_samples(self):
        gen = ADSBGenerator(seed=42)
        with pytest.raises(ValueError, match="No data to export"):
            gen.export("/tmp/test.csv", samples=[])

    def test_raises_valueerror_unsupported_extension(self, tmp_path):
        gen = ADSBGenerator(seed=42)
        samples = gen.generate(1)
        with pytest.raises(ValueError, match="Unsupported format"):
            gen.export(str(tmp_path / "test.xml"), samples=samples)


class TestExportNPZ:
    def setup_method(self):
        self.gen = ADSBGenerator(seed=42)

    def test_creates_file(self, tmp_path):
        samples = self.gen.generate(1)
        self.gen.export(str(tmp_path / "out.npz"), samples=samples)
        assert (tmp_path / "out.npz").exists()

    def test_loads_with_numpy(self, tmp_path):
        samples = self.gen.generate(2)
        self.gen.export(str(tmp_path / "out.npz"), samples=samples)
        loaded = np.load(str(tmp_path / "out.npz"), allow_pickle=True)
        assert "sample_0" in loaded
        assert "sample_1" in loaded

    def test_preserves_clean_signal(self, tmp_path):
        samples = self.gen.generate(1)
        self.gen.export(str(tmp_path / "out.npz"), samples=samples)
        loaded = np.load(str(tmp_path / "out.npz"), allow_pickle=True)
        sample_dict = loaded["sample_0"].item()
        np.testing.assert_array_equal(sample_dict["clean_signal"], samples[0].clean_signal)

    def test_preserves_channel_signal(self, tmp_path):
        samples = self.gen.generate(1)
        self.gen.export(str(tmp_path / "out.npz"), samples=samples)
        loaded = np.load(str(tmp_path / "out.npz"), allow_pickle=True)
        sample_dict = loaded["sample_0"].item()
        np.testing.assert_array_equal(sample_dict["channel_signal"], samples[0].channel_signal)

    def test_preserves_message(self, tmp_path):
        samples = self.gen.generate(1)
        self.gen.export(str(tmp_path / "out.npz"), samples=samples)
        loaded = np.load(str(tmp_path / "out.npz"), allow_pickle=True)
        sample_dict = loaded["sample_0"].item()
        assert sample_dict["message"] == samples[0].message

    def test_multiple_samples(self, tmp_path):
        samples = self.gen.generate(5)
        self.gen.export(str(tmp_path / "out.npz"), samples=samples)
        loaded = np.load(str(tmp_path / "out.npz"), allow_pickle=True)
        for i in range(5):
            assert f"sample_{i}" in loaded


class TestExportCSV:
    def setup_method(self):
        self.gen = ADSBGenerator(seed=42)

    def test_creates_file(self, tmp_path):
        samples = self.gen.generate(1)
        self.gen.export(str(tmp_path / "out.csv"), samples=samples)
        assert (tmp_path / "out.csv").exists()

    def test_has_header_row(self, tmp_path):
        samples = self.gen.generate(1)
        self.gen.export(str(tmp_path / "out.csv"), samples=samples)
        with open(tmp_path / "out.csv", "r") as f:
            reader = csv.reader(f)
            headers = next(reader)
            assert len(headers) > 0

    def test_header_starts_with_sample_number(self, tmp_path):
        samples = self.gen.generate(1)
        self.gen.export(str(tmp_path / "out.csv"), samples=samples)
        with open(tmp_path / "out.csv", "r") as f:
            reader = csv.reader(f)
            headers = next(reader)
            assert headers[0] == "sample_number"

    def test_excludes_signal_columns(self, tmp_path):
        samples = self.gen.generate(1)
        self.gen.export(str(tmp_path / "out.csv"), samples=samples)
        with open(tmp_path / "out.csv", "r") as f:
            reader = csv.reader(f)
            headers = next(reader)
            assert "clean_signal" not in headers
            assert "channel_signal" not in headers

    def test_includes_tx_params_columns(self, tmp_path):
        samples = self.gen.generate(1)
        self.gen.export(str(tmp_path / "out.csv"), samples=samples)
        with open(tmp_path / "out.csv", "r") as f:
            reader = csv.reader(f)
            headers = next(reader)
            assert "amplitude" in headers

    def test_includes_channel_params_columns(self, tmp_path):
        samples = self.gen.generate(1)
        self.gen.export(str(tmp_path / "out.csv"), samples=samples)
        with open(tmp_path / "out.csv", "r") as f:
            reader = csv.reader(f)
            headers = next(reader)
            assert "snr_db" in headers

    def test_row_count_matches_samples(self, tmp_path):
        samples = self.gen.generate(3)
        self.gen.export(str(tmp_path / "out.csv"), samples=samples)
        with open(tmp_path / "out.csv", "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert len(rows) == 4

    def test_sample_number_values(self, tmp_path):
        samples = self.gen.generate(3)
        self.gen.export(str(tmp_path / "out.csv"), samples=samples)
        with open(tmp_path / "out.csv", "r") as f:
            reader = csv.reader(f)
            next(reader)
            for i, row in enumerate(reader):
                assert row[0] == f"sample_{i}"

    def test_message_value_matches(self, tmp_path):
        samples = self.gen.generate(1)
        self.gen.export(str(tmp_path / "out.csv"), samples=samples)
        with open(tmp_path / "out.csv", "r") as f:
            reader = csv.reader(f)
            headers = next(reader)
            row = next(reader)
            msg_idx = headers.index("message")
            assert int(row[msg_idx]) == samples[0].message


class TestExportJSON:
    def setup_method(self):
        self.gen = ADSBGenerator(seed=42)

    def test_creates_file(self, tmp_path):
        samples = self.gen.generate(1)
        self.gen.export(str(tmp_path / "out.json"), samples=samples)
        assert (tmp_path / "out.json").exists()

    def test_loads_as_valid_json(self, tmp_path):
        samples = self.gen.generate(1)
        self.gen.export(str(tmp_path / "out.json"), samples=samples)
        with open(tmp_path / "out.json", "r") as f:
            data = json.load(f)
        assert isinstance(data, list)

    def test_array_length_matches_samples(self, tmp_path):
        samples = self.gen.generate(3)
        self.gen.export(str(tmp_path / "out.json"), samples=samples)
        with open(tmp_path / "out.json", "r") as f:
            data = json.load(f)
        assert len(data) == 3

    def test_excludes_clean_signal(self, tmp_path):
        samples = self.gen.generate(1)
        self.gen.export(str(tmp_path / "out.json"), samples=samples)
        with open(tmp_path / "out.json", "r") as f:
            data = json.load(f)
        sample_obj = data[0]["sample_0"]
        assert "clean_signal" not in sample_obj

    def test_excludes_channel_signal(self, tmp_path):
        samples = self.gen.generate(1)
        self.gen.export(str(tmp_path / "out.json"), samples=samples)
        with open(tmp_path / "out.json", "r") as f:
            data = json.load(f)
        sample_obj = data[0]["sample_0"]
        assert "channel_signal" not in sample_obj

    def test_includes_message(self, tmp_path):
        samples = self.gen.generate(1)
        self.gen.export(str(tmp_path / "out.json"), samples=samples)
        with open(tmp_path / "out.json", "r") as f:
            data = json.load(f)
        assert data[0]["sample_0"]["message"] == samples[0].message

    def test_includes_message_type(self, tmp_path):
        samples = self.gen.generate(1)
        self.gen.export(str(tmp_path / "out.json"), samples=samples)
        with open(tmp_path / "out.json", "r") as f:
            data = json.load(f)
        assert data[0]["sample_0"]["message_type"] == samples[0].message_type.value

    def test_includes_tx_params(self, tmp_path):
        samples = self.gen.generate(1)
        self.gen.export(str(tmp_path / "out.json"), samples=samples)
        with open(tmp_path / "out.json", "r") as f:
            data = json.load(f)
        assert "tx_params" in data[0]["sample_0"]

    def test_includes_channel_params(self, tmp_path):
        samples = self.gen.generate(1)
        self.gen.export(str(tmp_path / "out.json"), samples=samples)
        with open(tmp_path / "out.json", "r") as f:
            data = json.load(f)
        assert "channel_params" in data[0]["sample_0"]

    def test_sample_keys_match(self, tmp_path):
        samples = self.gen.generate(3)
        self.gen.export(str(tmp_path / "out.json"), samples=samples)
        with open(tmp_path / "out.json", "r") as f:
            data = json.load(f)
        for i in range(3):
            assert f"sample_{i}" in data[i]

    def test_compact_format_no_whitespace(self, tmp_path):
        samples = self.gen.generate(1)
        self.gen.export(str(tmp_path / "out.json"), samples=samples)
        with open(tmp_path / "out.json", "r") as f:
            content = f.read()
        assert "  " not in content
        assert "\n" not in content


class TestExportJSONL:
    def setup_method(self):
        self.gen = ADSBGenerator(seed=42)

    def test_creates_file(self, tmp_path):
        samples = self.gen.generate(1)
        self.gen.export(str(tmp_path / "out.jsonl"), samples=samples)
        assert (tmp_path / "out.jsonl").exists()

    def test_line_count_matches_samples(self, tmp_path):
        samples = self.gen.generate(3)
        self.gen.export(str(tmp_path / "out.jsonl"), samples=samples)
        with open(tmp_path / "out.jsonl", "r") as f:
            lines = f.readlines()
        assert len(lines) == 3

    def test_each_line_is_valid_json(self, tmp_path):
        samples = self.gen.generate(3)
        self.gen.export(str(tmp_path / "out.jsonl"), samples=samples)
        with open(tmp_path / "out.jsonl", "r") as f:
            for line in f:
                obj = json.loads(line)
                assert isinstance(obj, dict)

    def test_excludes_clean_signal(self, tmp_path):
        samples = self.gen.generate(1)
        self.gen.export(str(tmp_path / "out.jsonl"), samples=samples)
        with open(tmp_path / "out.jsonl", "r") as f:
            line = f.readline()
        obj = json.loads(line)
        sample_obj = list(obj.values())[0]
        assert "clean_signal" not in sample_obj

    def test_excludes_channel_signal(self, tmp_path):
        samples = self.gen.generate(1)
        self.gen.export(str(tmp_path / "out.jsonl"), samples=samples)
        with open(tmp_path / "out.jsonl", "r") as f:
            line = f.readline()
        obj = json.loads(line)
        sample_obj = list(obj.values())[0]
        assert "channel_signal" not in sample_obj

    def test_includes_message(self, tmp_path):
        samples = self.gen.generate(1)
        self.gen.export(str(tmp_path / "out.jsonl"), samples=samples)
        with open(tmp_path / "out.jsonl", "r") as f:
            line = f.readline()
        obj = json.loads(line)
        sample_obj = list(obj.values())[0]
        assert sample_obj["message"] == samples[0].message

    def test_sample_keys_match(self, tmp_path):
        samples = self.gen.generate(3)
        self.gen.export(str(tmp_path / "out.jsonl"), samples=samples)
        with open(tmp_path / "out.jsonl", "r") as f:
            for i, line in enumerate(f):
                obj = json.loads(line)
                assert f"sample_{i}" in obj

    def test_each_line_ends_with_newline(self, tmp_path):
        samples = self.gen.generate(2)
        self.gen.export(str(tmp_path / "out.jsonl"), samples=samples)
        with open(tmp_path / "out.jsonl", "r") as f:
            content = f.read()
        assert content.endswith("\n")


class TestExportBufferClear:
    def setup_method(self):
        self.gen = ADSBGenerator(seed=42)

    def test_clear_true_clears_buffer(self, tmp_path):
        self.gen.start_buffering()
        self.gen.generate(3)
        self.gen.export(str(tmp_path / "out.npz"), clear=True)
        assert len(self.gen._buffer) == 0

    def test_clear_false_preserves_buffer(self, tmp_path):
        self.gen.start_buffering()
        self.gen.generate(3)
        self.gen.export(str(tmp_path / "out.npz"), clear=False)
        assert len(self.gen._buffer) == 3

    def test_default_clears_buffer(self, tmp_path):
        self.gen.start_buffering()
        self.gen.generate(3)
        self.gen.export(str(tmp_path / "out.npz"))
        assert len(self.gen._buffer) == 0

    def test_clear_true_clears_buffer_even_with_explicit_samples(self, tmp_path):
        self.gen.start_buffering()
        self.gen.generate(3)
        samples = self.gen.generate(2)
        self.gen.export(str(tmp_path / "out.npz"), samples=samples, clear=True)
        assert len(self.gen._buffer) == 0


class TestExportBuffering:
    def test_start_buffering_enables_buffering(self):
        gen = ADSBGenerator(seed=42)
        gen.start_buffering()
        assert gen._buffering is True

    def test_stop_buffering_disables_buffering(self):
        gen = ADSBGenerator(seed=42)
        gen.start_buffering()
        gen.stop_buffering()
        assert gen._buffering is False

    def test_generate_returns_none_when_buffering(self):
        gen = ADSBGenerator(seed=42)
        gen.start_buffering()
        result = gen.generate(1)
        assert result is None

    def test_generate_returns_list_when_not_buffering(self):
        gen = ADSBGenerator(seed=42)
        result = gen.generate(1)
        assert isinstance(result, list)

    def test_samples_accumulate_in_buffer(self):
        gen = ADSBGenerator(seed=42)
        gen.start_buffering()
        gen.generate(3)
        gen.generate(2)
        assert len(gen._buffer) == 5

    def test_stop_buffering_clears_buffer_by_default(self):
        gen = ADSBGenerator(seed=42)
        gen.start_buffering()
        gen.generate(3)
        gen.stop_buffering()
        assert len(gen._buffer) == 0

    def test_stop_buffering_clear_false_preserves_buffer(self):
        gen = ADSBGenerator(seed=42)
        gen.start_buffering()
        gen.generate(3)
        gen.stop_buffering(clear=False)
        assert len(gen._buffer) == 3

    def test_export_from_buffer(self, tmp_path):
        gen = ADSBGenerator(seed=42)
        gen.start_buffering()
        gen.generate(3)
        gen.export(str(tmp_path / "out.npz"))
        assert (tmp_path / "out.npz").exists()
        assert len(gen._buffer) == 0

    def test_buffer_contains_adsbsample_instances(self):
        gen = ADSBGenerator(seed=42)
        gen.start_buffering()
        gen.generate(3)
        for sample in gen._buffer:
            assert isinstance(sample, ADSBSample)


class TestExportPathConversion:
    def test_accepts_string_path(self, tmp_path):
        gen = ADSBGenerator(seed=42)
        samples = gen.generate(1)
        gen.export(str(tmp_path / "out.npz"), samples=samples)
        assert (tmp_path / "out.npz").exists()

    def test_accepts_path_object(self, tmp_path):
        gen = ADSBGenerator(seed=42)
        samples = gen.generate(1)
        gen.export(tmp_path / "out.npz", samples=samples)
        assert (tmp_path / "out.npz").exists()


class TestExportRoundTrip:
    def test_npz_preserves_all_data(self, tmp_path):
        gen = ADSBGenerator(seed=42)
        samples = gen.generate(2)
        gen.export(str(tmp_path / "out.npz"), samples=samples)
        loaded = np.load(str(tmp_path / "out.npz"), allow_pickle=True)
        for i, sample in enumerate(samples):
            d = loaded[f"sample_{i}"].item()
            assert d["message"] == sample.message
            assert d["message_type"] == sample.message_type.value
            np.testing.assert_array_equal(d["clean_signal"], sample.clean_signal)
            np.testing.assert_array_equal(d["channel_signal"], sample.channel_signal)

    def test_csv_round_trip_preserves_scalar_fields(self, tmp_path):
        gen = ADSBGenerator(seed=42)
        samples = gen.generate(1)
        gen.export(str(tmp_path / "out.csv"), samples=samples)
        with open(tmp_path / "out.csv", "r") as f:
            reader = csv.reader(f)
            headers = next(reader)
            row = next(reader)
        msg_idx = headers.index("message")
        assert int(row[msg_idx]) == samples[0].message

    def test_json_round_trip_preserves_scalar_fields(self, tmp_path):
        gen = ADSBGenerator(seed=42)
        samples = gen.generate(1)
        gen.export(str(tmp_path / "out.json"), samples=samples)
        with open(tmp_path / "out.json", "r") as f:
            data = json.load(f)
        d = data[0]["sample_0"]
        assert d["message"] == samples[0].message
        assert d["message_type"] == samples[0].message_type.value

    def test_jsonl_round_trip_preserves_scalar_fields(self, tmp_path):
        gen = ADSBGenerator(seed=42)
        samples = gen.generate(1)
        gen.export(str(tmp_path / "out.jsonl"), samples=samples)
        with open(tmp_path / "out.jsonl", "r") as f:
            line = f.readline()
        d = json.loads(line)["sample_0"]
        assert d["message"] == samples[0].message
        assert d["message_type"] == samples[0].message_type.value
