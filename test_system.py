import io
import os
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from database import FEATURE_NAMES, connect, load_dataset, load_stats, resolve_audio_path
from features import (FEATURE_GROUPS, calculate_delta, calculate_spectrum, extract_features,
                      fit_normalizer, frame_signal, mel_filterbank, normalize, read_audio, spectral_features)
from evaluate import deduplicate, known_split, metrics, recording_group, unknown_split
from search import rank_candidates, search_audio


def wav_bytes(samples, sample_rate=16000):
    stream = io.BytesIO()
    sf.write(stream, samples, sample_rate, format="WAV", subtype="PCM_16")
    return stream.getvalue()


def synthetic_voice(sample_rate=16000, seconds=1.2):
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    return 0.2 * np.sin(2 * np.pi * 120 * t) + 0.05 * np.sin(2 * np.pi * 240 * t)


class FeatureTests(unittest.TestCase):
    def test_feature_contract(self):
        self.assertEqual(len(FEATURE_NAMES), 65)
        self.assertEqual(len(set(FEATURE_NAMES)), 65)
        self.assertEqual(FEATURE_NAMES[:5], ("rms_mean", "rms_std", "zcr_mean", "zcr_std", "silence_ratio"))
        self.assertEqual(FEATURE_NAMES[26:30], ("mfcc_0_mean", "mfcc_0_std", "mfcc_1_mean", "mfcc_1_std"))
        self.assertEqual(FEATURE_NAMES[-1], "delta_12_std")
        self.assertEqual(list(FEATURE_GROUPS.values()), [(0, 5), (5, 18), (18, 26), (26, 65)])

    def test_extract_real_waveform_statistics(self):
        source = wav_bytes(synthetic_voice())
        samples, _ = read_audio(source)
        result = extract_features(source, include_frames=True)
        self.assertEqual(result["vector"].shape, (65,))
        self.assertGreaterEqual(result["voiced_duration_sec"], 0.5)
        self.assertAlmostEqual(result["vector"][5], 120, delta=1)
        frames = np.lib.stride_tricks.sliding_window_view(samples, 400)[::160]
        rms = np.sqrt(np.mean(frames**2, axis=1))
        zcr = np.sum(frames[:, :-1] * frames[:, 1:] < 0, axis=1) / 399
        np.testing.assert_allclose(result["vector"][:5], [rms.mean(), rms.std(), zcr.mean(), zcr.std(), (rms <= .01).mean()])
        self.assertIn("frames", result)

    def test_stereo_resampling(self):
        mono = synthetic_voice(24000)
        samples, info = read_audio(wav_bytes(np.column_stack([mono, mono]), 24000))
        self.assertEqual(info["original_sample_rate"], 24000)
        self.assertEqual(info["original_channels"], 2)
        self.assertEqual(samples.ndim, 1)
        self.assertEqual(len(samples), 19200)

    def test_hash_ignores_container_format(self):
        samples = synthetic_voice()
        wav = wav_bytes(samples)
        decoded, _ = sf.read(io.BytesIO(wav))
        stream = io.BytesIO()
        sf.write(stream, decoded, 16000, format="FLAC", subtype="PCM_16")
        _, wav_info = read_audio(wav)
        _, flac_info = read_audio(stream.getvalue())
        self.assertEqual(wav_info["audio_hash"], flac_info["audio_hash"])

    def test_invalid_inputs(self):
        for source in (b"not audio", wav_bytes(np.zeros(16000)), wav_bytes(np.ones(1600) * .1)):
            with self.subTest(size=len(source)), self.assertRaises(ValueError):
                extract_features(source)

    def test_constant_unvoiced_audio_is_rejected(self):
        with self.assertRaises(ValueError):
            extract_features(wav_bytes(np.ones(32000) * .1))

    def test_path_traversal(self):
        with self.assertRaises(ValueError):
            resolve_audio_path("..\\outside.wav")
        self.assertTrue(resolve_audio_path("Dataset voices\\dataset_final\\test.wav").name == "test.wav")


class SpectrumTests(unittest.TestCase):
    def test_power_weighting_and_flux_adjacency(self):
        power = np.zeros((4, 257))
        power[0, 16], power[0, 48] = .75, .25
        power[2:, 32] = 1
        result = spectral_features(power)
        self.assertAlmostEqual(result["centroid"][0], 750)
        self.assertAlmostEqual(result["bandwidth"][0], np.sqrt(187500))
        self.assertEqual(result["rolloff"][0], 1500)
        self.assertTrue(np.isnan(result["flux"][:2]).all())
        self.assertEqual(result["flux"][2], 0)
        np.testing.assert_allclose(result["centroid"], spectral_features(power * 9)["centroid"])

    def test_framing_fft_and_mel(self):
        frames = frame_signal(np.ones(1000))
        self.assertEqual(frames.shape, (4, 400))
        self.assertEqual(calculate_spectrum(frames).shape, (4, 257))
        bank = mel_filterbank()
        self.assertEqual(bank.shape, (40, 257))
        self.assertTrue(((bank >= 0) & (bank <= 1)).all())
        self.assertGreater(np.ptp(bank.sum(axis=1)), 1)

    def test_delta_edge_repetition(self):
        np.testing.assert_allclose(calculate_delta(np.arange(5.0)), [.5, .8, 1, .8, .5])
        np.testing.assert_array_equal(calculate_delta(np.ones((5, 13))), np.zeros((5, 13)))


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.records = []
        for source, count in (("librispeech", 6), ("voxcelebrity", 3)):
            for person in range(count):
                for group in range(2):
                    for clip in range(2):
                        name = f"{person}-{group}-{clip}.wav" if source == "librispeech" else f"id{person}_video_{group}_{clip}.wav"
                        self.records.append({"source": source, "speaker_id": str(person), "file_path": name,
                                             "id": len(self.records), "audio_hash": str(len(self.records))})

    def test_disjoint_and_reproducible_splits(self):
        split = known_split(self.records)
        self.assertEqual(split, known_split(self.records))
        self.assertEqual(len(split["query_indices"]), 18)
        self.assertEqual(len(split["gallery_indices"]), 18)
        unknown = unknown_split(self.records)
        self.assertEqual(len(unknown["heldout_speakers"]), 6)
        self.assertEqual(recording_group({"source": "voxcelebrity", "file_path": "id1_a_b_c_00001.wav"}), "id1_a_b_c")

    def test_duplicates_and_conflicting_labels(self):
        rows = [dict(row) for row in self.records[:5]]
        rows[1]["audio_hash"] = rows[0]["audio_hash"]
        kept, audit = deduplicate(rows)
        self.assertEqual(len(kept), 4)
        self.assertEqual(audit["same_identity_duplicate_files_removed"], 1)
        rows[4]["audio_hash"] = rows[0]["audio_hash"]
        kept, audit = deduplicate(rows)
        self.assertEqual(audit["conflicting_identity_files_removed"], 3)
        self.assertEqual(len(kept), 2)

    def test_metric_definitions(self):
        query = {"source": "librispeech", "speaker_id": "0"}
        other = {"source": "librispeech", "speaker_id": "1"}
        self.assertEqual(metrics(query, [other, query, query]), {"top1": 0, "hit5": 1, "precision5": .4, "mrr5": .5})
        self.assertEqual(metrics(query, []), {"top1": 0, "hit5": 0, "precision5": 0, "mrr5": 0})


class NormalizerTests(unittest.TestCase):
    def setUp(self):
        self.matrix = np.arange(4 * 65, dtype=float).reshape(4, 65)

    def test_imputation_and_zscore(self):
        self.matrix[0, 1] = np.nan
        stats = fit_normalizer(self.matrix)
        self.assertEqual(stats["median"][1], np.median(self.matrix[1:, 1]))
        normalized = normalize(self.matrix, stats)
        np.testing.assert_allclose(normalized.mean(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(normalized.std(axis=0), 1, atol=1e-12)
        self.assertTrue(np.isfinite(normalized).all())

    def test_query_does_not_change_stats(self):
        stats = fit_normalizer(self.matrix)
        original = {key: value.copy() for key, value in stats.items()}
        query = np.full(65, 1e6)
        normalize(query, stats)
        for key in original:
            np.testing.assert_array_equal(stats[key], original[key])

    def test_zero_variance_query_is_inactive(self):
        self.matrix[:, 0] = 3
        stats = fit_normalizer(self.matrix)
        query = np.zeros(65)
        query[0] = 999
        self.assertEqual(normalize(query, stats)[0], 0)
        self.assertTrue((normalize(self.matrix, stats)[:, 0] == 0).all())

    def test_all_missing_dimension_fails(self):
        self.matrix[:, 3] = np.nan
        with self.assertRaises(ValueError):
            fit_normalizer(self.matrix)


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.records = [{"id": i, "file_path": name, "audio_hash": str(i)}
                        for i, name in enumerate(["b.wav", "a.wav", "c.wav"])]
        self.matrix = np.ones((3, 65))

    def test_score_tie_and_contributions(self):
        results = rank_candidates(np.zeros(65), self.matrix, self.records, explain=True)
        self.assertEqual([r["file_path"] for r in results], ["a.wav", "b.wav", "c.wav"])
        self.assertAlmostEqual(results[0]["distance"], np.sqrt(65))
        self.assertAlmostEqual(results[0]["similarity"], 1 / (1 + np.sqrt(65)))
        self.assertEqual(results[0]["group_contributions"], {"time": 5, "voice": 13, "spectral": 8, "cepstral": 39})
        self.assertAlmostEqual(sum(results[0]["group_contributions"].values()), results[0]["distance"]**2)
        self.assertEqual(len(results[0]["feature_differences"]), 65)

    def test_self_match_and_baseline(self):
        result = rank_candidates(np.zeros(65), self.matrix, self.records, exclude_hash="1", indices=range(26, 65))
        self.assertEqual(len(result), 2)
        self.assertNotIn("1", [r["audio_hash"] for r in result])
        self.assertAlmostEqual(result[0]["distance"], np.sqrt(39))

    def test_query_service_uses_saved_stats(self):
        gallery = np.arange(195, dtype=float).reshape(3, 65)
        stats = fit_normalizer(gallery)
        extracted = {"vector": gallery[0], "audio_hash": "0", "duration_sec": 6,
                     "original_sample_rate": 16000, "original_channels": 1,
                     "voiced_duration_sec": 4, "warnings": []}
        with patch("search.extract_features", return_value=extracted), \
             patch("search.load_dataset", return_value=(self.records, gallery)), \
             patch("search.load_stats", return_value=stats):
            result = search_audio(b"mock")
        self.assertEqual(result["excluded_self_count"], 1)
        self.assertEqual(len(result["query_features"]), 65)
        self.assertEqual(result["candidates_count"], 2)
        self.assertTrue(all(r["audio_hash"] != "0" for r in result["results"]))


@unittest.skipUnless(os.getenv("RUN_MYSQL_TESTS") == "1", "Đặt RUN_MYSQL_TESTS=1 sau khi xây CSDL.")
class MySQLReadOnlyTests(unittest.TestCase):
    def test_schema_and_data(self):
        with connect() as conn, conn.cursor() as cursor:
            cursor.execute("SHOW TABLES")
            self.assertEqual({row[0] for row in cursor.fetchall()}, {"audio_files", "audio_features", "feature_stats"})
            cursor.execute("SELECT COUNT(*) FROM feature_stats")
            self.assertEqual(cursor.fetchone()[0], 65)
            cursor.execute("SELECT COUNT(*) FROM audio_features v LEFT JOIN audio_files a ON a.id=v.audio_id WHERE a.id IS NULL")
            self.assertEqual(cursor.fetchone()[0], 0)
        records, matrix = load_dataset()
        self.assertGreaterEqual(len(records), 1000)
        self.assertEqual(matrix.shape, (len(records), 65))
        self.assertTrue(np.isfinite(normalize(matrix, load_stats())).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
