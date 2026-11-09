import argparse
import csv
import json
from pathlib import Path
import time

import mysql.connector
import numpy as np

from database import RESULTS_DIR, load_dataset, load_stats
from features import FEATURE_GROUPS, FEATURE_NAMES, FEATURE_UNITS, extract_features, normalize


def rank_candidates(query_z, matrix_z, records, k=5, exclude_hash=None, indices=None, explain=False):
    """Tính khoảng cách Euclid trên vector z-score và trả k ứng viên gần nhất."""
    if k < 1:
        raise ValueError("Số kết quả phải lớn hơn 0.")
    selected = np.arange(len(FEATURE_NAMES)) if indices is None else np.asarray(indices, dtype=int)
    query_z, matrix_z = np.asarray(query_z), np.asarray(matrix_z)
    differences = matrix_z[:, selected] - query_z[selected]
    distances = np.sqrt(np.sum(differences * differences, axis=1))
    eligible = [i for i, row in enumerate(records) if not exclude_hash or row["audio_hash"] != exclude_hash]
    nearest = sorted(eligible, key=lambda i: (float(distances[i]), records[i]["file_path"]))[:k]
    results = []
    selected_set = set(selected.tolist())
    for rank, index in enumerate(nearest, 1):
        distance = float(distances[index])
        item = {**records[index], "rank": rank, "distance": distance, "similarity": 1.0 / (1.0 + distance)}
        if explain:
            delta = query_z - matrix_z[index]
            squared = delta * delta
            item["group_contributions"] = {
                group: float(sum(squared[j] for j in range(start, stop) if j in selected_set))
                for group, (start, stop) in FEATURE_GROUPS.items()
            }
            item["feature_differences"] = [
                {"feature": FEATURE_NAMES[j], "query_z": float(query_z[j]),
                 "candidate_z": float(matrix_z[index, j]), "difference": float(delta[j]),
                 "squared_difference": float(squared[j])}
                for j in selected
            ]
        results.append(item)
    return results


def search_audio(source, k=5):
    """Luồng tìm kiếm đầy đủ: trích query, nạp gallery, chuẩn hóa, xếp hạng top 5."""
    start = time.perf_counter()
    extracted = extract_features(source)
    extraction_seconds = time.perf_counter() - start
    loading_start = time.perf_counter()
    records, raw_matrix = load_dataset()
    stats = load_stats()
    load_seconds = time.perf_counter() - loading_start
    comparison_start = time.perf_counter()
    raw = extracted["vector"]
    query_z = normalize(raw, stats)
    gallery_z = normalize(raw_matrix, stats)
    ranked = rank_candidates(query_z, gallery_z, records, k=k, exclude_hash=extracted["audio_hash"], explain=True)
    comparison_seconds = time.perf_counter() - comparison_start
    imputed = np.where(np.isnan(raw), stats["median"], raw)
    groups = {j: group for group, (begin, end) in FEATURE_GROUPS.items() for j in range(begin, end)}
    excluded = sum(row["audio_hash"] == extracted["audio_hash"] for row in records)
    return {
        "query_info": {key: extracted[key] for key in (
            "duration_sec", "original_sample_rate", "original_channels", "voiced_duration_sec", "warnings")},
        "query_hash": extracted["audio_hash"],
        "query_features": [
            {"feature": name, "group": groups[j], "unit": FEATURE_UNITS[j],
             "raw": None if np.isnan(raw[j]) else float(raw[j]),
             "imputed": float(imputed[j]), "z": float(query_z[j])}
            for j, name in enumerate(FEATURE_NAMES)
        ],
        "normalizer": {key: stats[key].tolist() for key in ("median", "mean", "std")},
        "results": ranked, "candidates_count": len(records) - excluded,
        "excluded_self_count": excluded,
        "timings": {"extraction_seconds": extraction_seconds, "load_seconds": load_seconds,
                    "comparison_seconds": comparison_seconds, "total_seconds": time.perf_counter() - start},
    }


def save_result(result, output):
    """Xuất kết quả tìm kiếm ra JSON và CSV để đưa vào báo cáo."""
    output = Path(output)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as stream:
        columns = ["rank", "file_path", "source", "speaker_id", "speaker_name", "distance", "similarity"]
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["results"])


def main():
    """CLI tìm kiếm một tệp âm thanh và lưu kết quả vào results/."""
    parser = argparse.ArgumentParser(description="Tìm 5 file giọng nói tương đồng từ MySQL.")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output", type=Path, help="Lưu kết quả JSON và CSV (thư mục phải tồn tại).")
    args = parser.parse_args()
    try:
        result = search_audio(args.audio)
        for item in result["results"]:
            print(f"{item['rank']}. {item['file_path']}  d={item['distance']:.6f}  s={item['similarity']:.6f}")
        RESULTS_DIR.mkdir(exist_ok=True)
        output = args.output or RESULTS_DIR / "query_result.json"
        save_result(result, output)
        print("Đã lưu:", output)
    except (ValueError, OSError, mysql.connector.Error) as exc:
        parser.exit(1, f"Lỗi: {exc}\n")


if __name__ == "__main__":
    main()
