import argparse
import csv
import json
import math
import os
from pathlib import Path, PureWindowsPath
import random
import time

import numpy as np


SEED = 42
K = 5
FEATURE_COUNT = 65
METRIC_NAMES = ("top1", "hit5", "precision5", "mrr5")
VARIANTS = {"full65": None, "cepstral39": tuple(range(26, 65))}


def speaker(row):
    """Identity is source-qualified; subset is deliberately not used."""
    values = (row.get("source"), row.get("speaker_id"))
    if any(value is None or not str(value).strip() for value in values):
        raise ValueError("Every record needs source and speaker_id.")
    return tuple(str(value) for value in values)


def _source_family(source):
    """Xác định họ nguồn libri/vox để chọn cách tách nhóm ghi."""
    value = str(source).lower()
    if value.startswith("libri"):
        return "libri"
    if value.startswith("vox"):
        return "vox"
    raise ValueError(f"Unsupported source: {source!r}")


def recording_group(row):
    """Libri chapter or Vox recording prefix, preserving video-ID underscores."""
    if not row.get("file_path"):
        raise ValueError("Every record needs file_path.")
    stem = PureWindowsPath(str(row["file_path"])).stem
    if _source_family(row["source"]) == "libri":
        parts = stem.split("-")
        if len(parts) < 3 or not all(parts[:3]):
            raise ValueError(f"Invalid Libri recording name: {stem!r}")
        return "-".join(parts[:2])
    parts = stem.rsplit("_", 1)
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Invalid Vox recording name: {stem!r}")
    return parts[0]


def _audio_hash(row):
    """Lấy hash âm thanh chuẩn hóa chữ thường, báo lỗi nếu thiếu."""
    value = row.get("audio_hash")
    if value is None or not str(value).strip():
        raise ValueError("Missing audio_hash: exact-duplicate exclusion is unverifiable.")
    return str(value).strip().lower()


def _order_key(row):
    """Khóa sắp xếp tất định theo đường dẫn, người nói và id bản ghi."""
    return (str(row.get("file_path", "")), *speaker(row), str(row.get("id", "")))


def _identity(identity):
    """Chuyển cặp (source, speaker_id) thành dict nhãn người nói."""
    return {"source": identity[0], "speaker_id": identity[1]}


def deduplicate(records):
    """Loại bản trùng hash cùng nhãn; nhãn mâu thuẫn thì loại toàn bộ nhóm."""
    buckets = {}
    for index, row in enumerate(records):
        speaker(row)
        buckets.setdefault(_audio_hash(row), []).append(index)
    kept, duplicate_groups, conflicts, excluded = [], [], [], []
    for audio_hash, indices in sorted(buckets.items()):
        indices = sorted(indices, key=lambda index: _order_key(records[index]))
        identities = sorted({speaker(records[index]) for index in indices})
        paths = [str(records[index].get("file_path", "")) for index in indices]
        if len(identities) > 1:
            conflicts.append({"audio_hash": audio_hash, "file_paths": paths,
                              "identities": [_identity(item) for item in identities]})
            excluded.extend({"index": index, "reason": "conflicting_hash_identity"}
                            for index in indices)
        else:
            kept.append(indices[0])
            if len(indices) > 1:
                duplicate_groups.append({"audio_hash": audio_hash, "kept_file_path": paths[0],
                                         "dropped_file_paths": paths[1:]})
                excluded.extend({"index": index, "reason": "same_identity_duplicate"}
                                for index in indices[1:])
    kept.sort(key=lambda index: _order_key(records[index]))
    report = {
        "input_files": len(records), "retained_files": len(kept),
        "same_identity_duplicate_groups": len(duplicate_groups),
        "same_identity_duplicate_files_removed": sum(len(g["dropped_file_paths"])
                                                      for g in duplicate_groups),
        "conflicting_hash_groups": len(conflicts),
        "conflicting_identity_files_removed": sum(len(g["file_paths"]) for g in conflicts),
        "duplicate_groups": duplicate_groups, "conflicts": conflicts, "excluded": excluded,
    }
    assert len({_audio_hash(records[i]) for i in kept}) == len(kept)
    return kept, report


def _speaker_groups(records):
    """Gom chỉ số bản ghi theo người nói rồi theo nhóm chương/video."""
    groups = {}
    for index, row in enumerate(records):
        groups.setdefault(speaker(row), {}).setdefault(recording_group(row), []).append(index)
    return groups


def _assert_split(records, split, unknown=False):
    """Kiểm tra split không trùng file/hash/nhóm ghi và đúng phạm vi người nói."""
    gallery, queries = set(split["gallery_indices"]), set(split["query_indices"])
    assert not gallery & queries, "Gallery/query file overlap"
    assert gallery | queries == set(range(len(records))), "Incomplete split membership"
    assert len({_audio_hash(row) for row in records}) == len(records), "Deduplicate before splitting"
    gallery_hashes = {_audio_hash(records[i]) for i in gallery}
    assert gallery_hashes.isdisjoint({_audio_hash(records[i]) for i in queries})
    gallery_groups = {(*speaker(records[i]), recording_group(records[i])) for i in gallery}
    query_groups = {(*speaker(records[i]), recording_group(records[i])) for i in queries}
    assert gallery_groups.isdisjoint(query_groups), "Recording-group leakage"
    gallery_speakers = {speaker(records[i]) for i in gallery}
    query_speakers = {speaker(records[i]) for i in queries}
    if unknown:
        heldout = {(item["source"], item["speaker_id"]) for item in split["heldout_speakers"]}
        assert query_speakers == heldout
        assert gallery_speakers.isdisjoint(heldout), "Unknown speaker leaked into gallery/fit"
    else:
        assert query_speakers <= gallery_speakers, "Known query has no gallery identity"
        for identity in query_speakers:
            assert len({group[2] for group in query_groups if group[:2] == identity}) == 1


def known_split(records, seed=SEED):
    """Chia query/gallery theo nhóm ghi, giữ một nhóm làm query cho người có >=2 nhóm."""
    rng = random.Random(seed)
    grouped = _speaker_groups(records)
    gallery, queries, eligible, single, heldout, counts = [], [], [], [], [], []
    for identity, groups in sorted(grouped.items()):
        names = sorted(groups)
        counts.append({**_identity(identity), "groups": len(names),
                       "files": sum(len(indices) for indices in groups.values())})
        selected = rng.choice(names) if len(names) >= 2 else None
        (eligible if selected is not None else single).append(_identity(identity))
        if selected is not None:
            heldout.append({**_identity(identity), "recording_group": selected})
        for name in names:
            (queries if name == selected else gallery).extend(groups[name])
    result = {
        "seed": seed, "gallery_indices": sorted(gallery, key=lambda i: _order_key(records[i])),
        "query_indices": sorted(queries, key=lambda i: _order_key(records[i])),
        "eligible_query_speakers": eligible, "excluded_one_group_speakers": single,
        "heldout_groups": heldout, "speaker_group_counts": counts,
    }
    _assert_split(records, result)
    return result


def unknown_split(records, seed=SEED):
    """Giữ toàn bộ 4 người Libri + 2 người Vox ngoài gallery làm thí nghiệm người lạ."""
    rng = random.Random(seed)
    identities = sorted({speaker(row) for row in records})
    heldout = set()
    for family, number in (("libri", 4), ("vox", 2)):
        available = [identity for identity in identities if _source_family(identity[0]) == family]
        if len(available) < number:
            raise ValueError(f"Unknown experiment requires {number} {family} speakers; found {len(available)}.")
        heldout.update(rng.sample(available, number))
    ordered = sorted(range(len(records)), key=lambda i: _order_key(records[i]))
    result = {
        "seed": seed, "heldout_speakers": [_identity(item) for item in sorted(heldout)],
        "gallery_indices": [i for i in ordered if speaker(records[i]) not in heldout],
        "query_indices": [i for i in ordered if speaker(records[i]) in heldout],
    }
    if not result["gallery_indices"]:
        raise ValueError("Unknown experiment leaves no gallery for fitting the normalizer.")
    _assert_split(records, result, unknown=True)
    return result


def metrics(query, candidates):
    """Binary same-identity retrieval metrics; precision always divides by 5."""
    identity = speaker(query)
    relevant = [speaker(row) == identity for row in candidates[:K]]
    first = next((rank for rank, match in enumerate(relevant, 1) if match), None)
    return {"top1": int(bool(relevant) and relevant[0]), "hit5": int(any(relevant)),
            "precision5": sum(relevant) / K, "mrr5": 1.0 / first if first else 0.0}


def _means(rows):
    """Trung bình các metric trên danh sách dòng kết quả."""
    return {name: float(np.mean([row[name] for row in rows])) if rows else None
            for name in METRIC_NAMES}


def _aggregate(rows):
    """Tổng hợp metric theo query (micro), người nói (macro) và từng nguồn."""
    grouped = {}
    for row in rows:
        grouped.setdefault(speaker(row), []).append(row)
    per_speaker = [{**_identity(identity), "queries": len(items), **_means(items)}
                   for identity, items in sorted(grouped.items())]
    by_source = {}
    for source in sorted({identity[0] for identity in grouped}):
        source_rows = [row for row in rows if row["source"] == source]
        speaker_rows = [row for row in per_speaker if row["source"] == source]
        by_source[source] = {"queries": len(source_rows), "speakers": len(speaker_rows),
                             "failed_queries": sum(row["status"] != "ok" for row in source_rows),
                             "query_micro": _means(source_rows), "speaker_macro": _means(speaker_rows)}
    return {"queries": len(rows), "speakers": len(per_speaker),
            "failed_queries": sum(row["status"] != "ok" for row in rows),
            "query_micro": _means(rows), "speaker_macro": _means(per_speaker),
            "per_speaker": per_speaker, "by_source": by_source}


def _distribution(values):
    """Thống kê phân phối min/mean/median/p95/max của dãy giá trị."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, "min": None, "mean": None, "median": None, "p95": None, "max": None}
    return {"count": len(array), "min": float(array.min()), "mean": float(array.mean()),
            "median": float(np.median(array)), "p95": float(np.percentile(array, 95)),
            "max": float(array.max())}


def _timing(values):
    """Tổng thời gian và phân phối thời gian mỗi truy vấn."""
    return {"total_seconds": float(sum(values)), "per_query_seconds": _distribution(values)}


def _clean_json(value):
    """Convert NumPy and nonfinite values before strict JSON serialization."""
    if isinstance(value, np.ndarray):
        return _clean_json(value.tolist())
    if isinstance(value, np.generic):
        return _clean_json(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, (Path, PureWindowsPath)):
        return str(value)
    return value


def _prepare(records, matrix):
    """Lọc bản ghi hợp lệ, khử trùng lặp toàn cục và kiểm tra đặc trưng trước đánh giá."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (len(records), FEATURE_COUNT):
        raise ValueError(f"Expected raw matrix ({len(records)}, {FEATURE_COUNT}), got {matrix.shape}.")
    hashable, failures = [], []
    for index, row in enumerate(records):
        try:
            speaker(row)
            _audio_hash(row)
        except (KeyError, TypeError, ValueError) as exc:
            failures.append({"index": index, "file_path": row.get("file_path"), "error": str(exc)})
        else:
            hashable.append(index)
    # Resolve conflicts before validation so malformed rows cannot hide another identity.
    kept, audit = deduplicate([records[index] for index in hashable])
    audit["excluded"] = [{**item, "index": hashable[item["index"]]} for item in audit["excluded"]]
    original_indices = []
    for kept_index in kept:
        index = hashable[kept_index]
        row = records[index]
        try:
            recording_group(row)
            if np.isinf(matrix[index]).any():
                raise ValueError("Infinite raw features; NaNs are allowed, infinities are not.")
        except (KeyError, TypeError, ValueError) as exc:
            failures.append({"index": index, "file_path": row.get("file_path"), "error": str(exc)})
        else:
            original_indices.append(index)
    if not original_indices:
        raise ValueError("No valid, nonconflicting hashed records remain for evaluation.")
    return ([records[index] for index in original_indices], matrix[original_indices],
            original_indices, audit, failures)


def _normalize_split(records, matrix, split, fit_normalizer, normalize):
    """Fit thống kê chỉ trên gallery rồi chuẩn hóa cả gallery lẫn query."""
    gallery_indices = split["gallery_indices"]
    if not gallery_indices:
        raise ValueError("Cannot fit a normalizer without gallery files.")
    start = time.perf_counter()
    stats = fit_normalizer(matrix[gallery_indices])
    normalized = normalize(matrix, stats)
    if normalized.shape != matrix.shape or not np.isfinite(normalized).all():
        raise ValueError("Normalizer must return finite normalized features of the original shape.")
    return (stats, normalized, normalized[gallery_indices],
            [records[index] for index in gallery_indices], time.perf_counter() - start)


def _rank(query, query_z, gallery_z, gallery, rank_candidates, indices=None, explain=False):
    """Gọi xếp hạng ứng viên, đo thời gian và ghi nhận lỗi nếu thất bại."""
    start = time.perf_counter()
    try:
        candidates = rank_candidates(query_z, gallery_z, gallery, k=K,
                                     exclude_hash=query["audio_hash"], indices=indices, explain=explain)
    except Exception as exc:
        # Count failed retrievals in the denominator instead of inflating the metrics.
        return [], time.perf_counter() - start, f"{type(exc).__name__}: {exc}"
    return candidates, time.perf_counter() - start, None


def _example(index, records, matrix, normalized, stats, gallery_z, gallery,
             rank_candidates, comparison=False):
    """Xuất ví dụ truy vấn đầy đủ: vector, thống kê, top 5 và đối chiếu cepstral."""
    query = records[index]
    candidates, _, error = _rank(query, normalized[index], gallery_z, gallery,
                                  rank_candidates, explain=True)
    result = {"query_record": dict(query), "recording_group": recording_group(query),
              "normalizer_stats": stats, "normalizer_fit_files": len(gallery),
              "query_raw": matrix[index], "query_normalized": normalized[index],
              "ranking_variant": "full65", "top5": candidates, "error": error}
    if comparison:
        candidates, _, error = _rank(query, normalized[index], gallery_z, gallery,
                                      rank_candidates, indices=VARIANTS["cepstral39"], explain=True)
        result["cepstral39_comparison"] = {"top5": candidates, "error": error}
    return result


def _coverage(records, split):
    """Thống kê phạm vi phủ của split known theo nguồn và người nói."""
    result = {"files": len(records), "speakers": len({speaker(row) for row in records}),
              "gallery_files": len(split["gallery_indices"]), "query_files": len(split["query_indices"]),
              "gallery_speakers": len({speaker(records[i]) for i in split["gallery_indices"]}),
              "eligible_query_speakers": len(split["eligible_query_speakers"]),
              "excluded_one_group_speakers": len(split["excluded_one_group_speakers"])}
    result["by_source"] = {}
    for source in sorted({row["source"] for row in records}):
        result["by_source"][source] = {
            "files": sum(row["source"] == source for row in records),
            "gallery_files": sum(records[i]["source"] == source for i in split["gallery_indices"]),
            "query_files": sum(records[i]["source"] == source for i in split["query_indices"]),
            "eligible_query_speakers": sum(row["source"] == source for row in split["eligible_query_speakers"]),
            "excluded_one_group_speakers": sum(row["source"] == source for row in split["excluded_one_group_speakers"]),
        }
    return result


def _membership(original, retained_indices, split, exclusions):
    """Ghi vai trò gallery/query/excluded của từng bản ghi gốc kèm lý do loại."""
    gallery = {retained_indices[i] for i in split["gallery_indices"]}
    queries = {retained_indices[i] for i in split["query_indices"]}
    rows = []
    for index in sorted(range(len(original)), key=lambda i: (str(original[i].get("file_path", "")), i)):
        record = original[index]
        try:
            group = recording_group(record)
        except (KeyError, TypeError, ValueError):
            group = None
        role = "gallery" if index in gallery else "query" if index in queries else "excluded"
        rows.append({"seed": split["seed"], "dataset_index": index, "role": role,
                     "exclusion_reason": exclusions.get(index, ""),
                     **{name: record.get(name) for name in ("id", "file_path", "source", "speaker_id",
                                                            "subset", "audio_hash")},
                     "recording_group": group})
    assert len(gallery) + len(queries) + len(exclusions) == len(original)
    return rows


def evaluate(records, matrix, seed=SEED):
    """Đánh giá known/unknown trong bộ nhớ, không ghi đè thống kê demo."""
    from features import FEATURE_GROUPS, FEATURE_NAMES, fit_normalizer, normalize
    from search import rank_candidates

    if len(FEATURE_NAMES) != FEATURE_COUNT:
        raise ValueError("Evaluation requires the approved 65-feature schema.")
    records = list(records)
    retained, raw, original_indices, duplicates, validation_failures = _prepare(records, matrix)
    known = known_split(retained, seed)
    unknown = unknown_split(retained, seed)
    coverage = _coverage(retained, known)
    warnings = []
    expected = {"eligible_query_speakers": 42, "excluded_one_group_speakers": 12}
    if any(coverage[name] != value for name, value in expected.items()):
        warnings.append("Observed known-speaker group counts differ from expected 42 eligible / 12 single-group; "
                        "check coverage, deduplication and failed/missing records. No split was forced to match.")

    stats, normalized, gallery_z, gallery, known_normalization_seconds = _normalize_split(
        retained, raw, known, fit_normalizer, normalize)
    query_rows, known_summaries, known_times, ranking_failures = [], {}, {}, []
    first_failure = None
    for variant, indices in VARIANTS.items():
        variant_rows, elapsed = [], []
        for index in known["query_indices"]:
            query = retained[index]
            candidates, seconds, error = _rank(query, normalized[index], gallery_z, gallery,
                                               rank_candidates, indices=indices)
            score = metrics(query, candidates)
            row = {"variant": variant, "dataset_index": original_indices[index],
                   **{name: query.get(name) for name in ("id", "file_path", "source", "speaker_id", "audio_hash")},
                   "recording_group": recording_group(query), "status": "failed" if error else "ok",
                   "error": error, **score, "candidate_count": len(candidates),
                   "relevant5": sum(speaker(item) == speaker(query) for item in candidates[:K]),
                   "ranking_seconds": seconds,
                   "top1_id": candidates[0].get("id") if candidates else None,
                   "top1_file_path": candidates[0].get("file_path") if candidates else None,
                   "top1_distance": candidates[0]["distance"] if candidates else None,
                   "top5_ids": [item.get("id") for item in candidates[:K]]}
            if error:
                ranking_failures.append({"experiment": "known", "variant": variant,
                                         "query_record": dict(query), "error": error})
            if variant == "full65" and not error and score["top1"] == 0 and first_failure is None:
                first_failure = index
            variant_rows.append(row)
            elapsed.append(seconds)
        query_rows.extend(variant_rows)
        known_summaries[variant] = _aggregate(variant_rows)
        known_times[variant] = _timing(elapsed)

    example_start = time.perf_counter()
    example_known = (_example(known["query_indices"][0], retained, raw, normalized, stats,
                              gallery_z, gallery, rank_candidates, comparison=True)
                     if known["query_indices"] else None)
    example_failure = (_example(first_failure, retained, raw, normalized, stats, gallery_z,
                                gallery, rank_candidates, comparison=True)
                       if first_failure is not None else None)
    known_example_seconds = time.perf_counter() - example_start

    unknown_stats, unknown_z, unknown_gallery_z, unknown_gallery, unknown_normalization_seconds = _normalize_split(
        retained, raw, unknown, fit_normalizer, normalize)
    heldout = {(row["source"], row["speaker_id"]) for row in unknown["heldout_speakers"]}
    assert all(speaker(row) not in heldout for row in unknown_gallery)
    unknown_times, nearest_distances, returned_distances, by_speaker = [], [], [], {}
    unknown_failures = 0
    for index in unknown["query_indices"]:
        query = retained[index]
        candidates, seconds, error = _rank(query, unknown_z[index], unknown_gallery_z,
                                           unknown_gallery, rank_candidates)
        unknown_times.append(seconds)
        bucket = by_speaker.setdefault(speaker(query), {"first_index": index, "queries": 0,
                                                       "failed_queries": 0, "nearest": [], "returned": [], "times": []})
        bucket["queries"] += 1
        bucket["times"].append(seconds)
        distances = [float(row["distance"]) for row in candidates[:K]]
        returned_distances.extend(distances)
        bucket["returned"].extend(distances)
        if distances:
            nearest_distances.append(distances[0])
            bucket["nearest"].append(distances[0])
        if error:
            unknown_failures += 1
            bucket["failed_queries"] += 1
            ranking_failures.append({"experiment": "unknown", "variant": "full65",
                                     "query_record": dict(query), "error": error})
    example_start = time.perf_counter()
    examples_unknown = [_example(bucket["first_index"], retained, raw, unknown_z, unknown_stats,
                                  unknown_gallery_z, unknown_gallery, rank_candidates)
                        for identity, bucket in sorted(by_speaker.items())]
    unknown_example_seconds = time.perf_counter() - example_start
    example_errors = []
    for label, example in [("known", example_known), ("known_failure", example_failure)] + [
            ("unknown", item) for item in examples_unknown]:
        if example is None:
            continue
        for variant, result in (("full65", example), ("cepstral39", example.get("cepstral39_comparison", {}))):
            if result.get("error"):
                example_errors.append({"example": label, "variant": variant,
                                       "query_record": example["query_record"], "error": result["error"]})

    exclusions = {item["index"]: item["reason"] for item in duplicates["excluded"]}
    exclusions.update({item["index"]: "validation_failure: " + item["error"] for item in validation_failures})
    limitations = [
        "Metrics measure retrieval of the same labelled (source, speaker_id), not acoustic similarity, "
        "perceptual voice quality, speaker-verification accuracy or calibrated probability.",
        f"No weights, thresholds or hyperparameters are tuned. This is one seed-{seed} split, not cross-validation.",
        "Groups are filename-derived chapters/videos; they do not prove channel/session independence.",
        "Known queries cover only speakers with at least two retained recording groups. Single-group "
        "speakers are gallery-only distractors; source and speaker-macro summaries expose unequal coverage.",
        "Unknown-speaker distances/candidates are manual comparisons only: no same-speaker accuracy, "
        "detection threshold, unknown-detection claim, or cross-normalizer distance comparability.",
        "load_dataset returns ready feature rows only. Failed extraction, unprocessed and missing database "
        "rows cannot be counted through this API; extraction failure counts are unknown, not zero.",
        "Ranking timings exclude extraction, normalization and explanation export; they are local wall-clock "
        "measurements without warm-up correction and are not a controlled speed benchmark.",
    ]
    report = {
        "schema_version": 1,
        "summary": {"status": "partial_failure" if ranking_failures or example_errors or validation_failures else "ok",
                    "loaded_files": len(records), "retained_files": len(retained),
                    "removed_files": len(records) - len(retained),
                    "eligible_query_speakers": coverage["eligible_query_speakers"],
                    "excluded_one_group_speakers": coverage["excluded_one_group_speakers"],
                    "known_query_files": coverage["query_files"], "unknown_heldout_speakers": len(heldout),
                    "unknown_query_files": len(unknown["query_indices"]),
                    "validation_failures": len(validation_failures),
                    "ranking_failures": len(ranking_failures), "example_failures": len(example_errors),
                    "extraction_failures": None},
        "settings": {"seed": seed, "rng": "stdlib random.Random; sorted speakers/groups/file paths",
                     "k": K, "feature_names": list(FEATURE_NAMES), "feature_groups": FEATURE_GROUPS,
                     "variants": {name: {"indices": list(range(FEATURE_COUNT)) if indices is None else list(indices),
                                          "distance": "unweighted Euclidean in gallery-normalized feature space"}
                                  for name, indices in VARIANTS.items()},
                     "identity": ["source", "speaker_id"], "source_subset_used": False,
                     "deduplication": "global audio_hash before either split; lexical-first same identity, drop all conflicts",
                     "known_split": "one random recording group per eligible speaker; all files in that group queried",
                     "unknown_split": {"libri_speakers": 4, "vox_speakers": 2, "query_all_heldout_files": True,
                                       "variant": "full65", "independent_rng_restart": True},
                     "normalization": "shared fit_normalizer/normalize; median, mean and std fitted ONLY on each experiment's gallery",
                     "normalizer_written_to_database": False,
                     "explanations": {"group_contributions": "shared ranker's additive squared-distance contributions",
                                      "feature_differences": "shared ranker's per-feature normalized differences; 65 for full65",
                                      "similarity": "shared ranker's heuristic ranking score, not a probability"},
                     "membership_dataset_index": "zero-based index in the original load_dataset records, including exclusions",
                     "metrics": {"top1": "first result has same source-qualified identity",
                                 "hit5": "any of up to five results has same identity",
                                 "precision5": "number of relevant top-five results / 5, even with fewer candidates",
                                 "mrr5": "reciprocal first relevant rank within five, else 0",
                                 "query_micro": "arithmetic mean across query files",
                                 "speaker_macro": "arithmetic mean of per-speaker query means",
                                 "ranking_failure_policy": "empty result, zero metrics; counted and detailed separately"},
                     "example_selection": "lexicographic first known query; first successfully ranked full65 top1 error; "
                                          "lexicographic first file of each heldout unknown speaker",
                     "expected_group_coverage": expected},
        "coverage": coverage, "deduplication": duplicates,
        "failures": {"validation": validation_failures, "ranking": ranking_failures,
                     "examples": example_errors, "extraction": None},
        "known": {"split": {key: value for key, value in known.items() if not key.endswith("_indices")},
                  "normalizer_stats": stats, "normalizer_fit_files": len(gallery), "metrics": known_summaries},
        "unknown": {"purpose": "manual comparisons only", "heldout_speakers": unknown["heldout_speakers"],
                    "gallery_files": len(unknown_gallery), "query_files": len(unknown["query_indices"]),
                    "failed_queries": unknown_failures, "normalizer_fit_files": len(unknown_gallery),
                    "normalizer_stats": unknown_stats, "nearest_distance": _distribution(nearest_distances),
                    "returned_top5_distance": _distribution(returned_distances),
                    "by_speaker": [{**_identity(identity), "queries": bucket["queries"],
                                    "failed_queries": bucket["failed_queries"],
                                    "nearest_distance": _distribution(bucket["nearest"]),
                                    "returned_top5_distance": _distribution(bucket["returned"]),
                                    "ranking_timing": _timing(bucket["times"])}
                                   for identity, bucket in sorted(by_speaker.items())]},
        "timing": {"feature_extraction": {"performed": False, "seconds": None},
                   "normalization_seconds": {"known": known_normalization_seconds, "unknown": unknown_normalization_seconds},
                   "ranking": {"known": known_times, "unknown_full65": _timing(unknown_times)},
                   "example_generation_seconds": {"known": known_example_seconds, "unknown": unknown_example_seconds}},
        "warnings": warnings, "metric_limitations": limitations,
        "examples": {"known": "example_known.json", "known_failure": "example_failure.json",
                     "known_failure_available": example_failure is not None,
                     "unknown": "example_unknown.json", "unknown_count": len(examples_unknown),
                     "absent_example_representation": "JSON null"},
        "files": {"report": "evaluation.json", "known_queries": "known_queries.csv",
                  "known_membership": "split_known.csv", "unknown_membership": "split_unknown.csv"},
    }
    return _clean_json({"evaluation": report, "known_queries": query_rows,
                        "split_known": _membership(records, original_indices, known, exclusions),
                        "split_unknown": _membership(records, original_indices, unknown, exclusions),
                        "example_known": example_known, "example_failure": example_failure,
                        "example_unknown": examples_unknown})


MEMBERSHIP_FIELDS = ("seed", "dataset_index", "role", "exclusion_reason", "id", "file_path", "source",
                     "speaker_id", "subset", "audio_hash", "recording_group")
QUERY_FIELDS = ("variant", "dataset_index", "id", "file_path", "source", "speaker_id", "audio_hash",
                "recording_group", "status", "error", "top1", "hit5", "precision5", "mrr5",
                "candidate_count", "relevant5", "ranking_seconds", "top1_id", "top1_file_path",
                "top1_distance", "top5_ids")


def write_results(payload, directory):
    """Write only the seven requested report artifacts; never database stats."""
    directory = Path(directory)
    # Validate all JSON before opening output files, including NaN conversion.
    json_text = {name: json.dumps(_clean_json(payload[name]), ensure_ascii=False,
                                  indent=2, allow_nan=False) + "\n"
                 for name in ("evaluation", "example_known", "example_failure", "example_unknown")}
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in json_text.items():
        (directory / f"{name}.json").write_text(text, encoding="utf-8")
    for name, fields in (("known_queries", QUERY_FIELDS), ("split_known", MEMBERSHIP_FIELDS),
                         ("split_unknown", MEMBERSHIP_FIELDS)):
        with (directory / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in payload[name]:
                row = _clean_json(row)
                writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                                 if isinstance(value, (list, dict)) else value for key, value in row.items()})


def main():
    """CLI đọc đặc trưng MySQL, chạy đánh giá seed 42 và xuất 7 artifact báo cáo."""
    parser = argparse.ArgumentParser(description="Read stored MySQL raw features and export seed-42 retrieval evaluation.")
    parser.parse_args()
    from database import RESULTS_DIR, ROOT, get_overview, load_dataset, load_stats
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    if not os.getenv("DB_USER", "").strip():
        parser.exit(1, "DB_USER is not explicitly configured; no MySQL connection attempted.\n")
    load_stats()
    start = time.perf_counter()
    records, matrix = load_dataset()
    load_seconds = time.perf_counter() - start
    payload = evaluate(records, matrix)
    overview = get_overview()
    failed = sum(row["files"] for row in overview if row["status"] == "failed")
    payload["evaluation"]["summary"]["extraction_failures"] = failed
    payload["evaluation"]["failures"]["extraction"] = {"failed_files": failed, "database_overview": overview}
    payload["evaluation"]["metric_limitations"] = [
        text for text in payload["evaluation"]["metric_limitations"] if not text.startswith("load_dataset returns")
    ]
    payload["evaluation"]["timing"]["database_read_seconds"] = load_seconds
    payload["evaluation"]["timing"]["evaluation_seconds_excluding_export"] = time.perf_counter() - start - load_seconds
    payload["evaluation"]["metadata"] = {"root": str(ROOT), "results_directory": str(RESULTS_DIR),
                                           "input": "MySQL load_dataset(): ready rows with stored raw features",
                                           "numpy_version": np.__version__}
    write_results(payload, RESULTS_DIR)
    print(json.dumps(payload["evaluation"]["summary"], ensure_ascii=False, indent=2, allow_nan=False))
    print(f"Reports: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
