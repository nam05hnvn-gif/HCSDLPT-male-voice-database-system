import argparse
import csv
import json
import os
from pathlib import Path, PureWindowsPath
import re
import time

from dotenv import load_dotenv
import mysql.connector
import numpy as np

from features import EXTRACTOR_SETTINGS, FEATURE_NAMES, extract_features, fit_normalizer

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Dataset voices" / "dataset_final"
MANIFEST = ROOT / "Dataset voices" / "dataset_manifest.csv"
RESULTS_DIR = ROOT / "results"


def connection_settings():
    """Đọc cấu hình MySQL từ .env/biến môi trường và validate tên database."""
    load_dotenv(ROOT / ".env")
    user = os.getenv("DB_USER", "").strip()
    name = os.getenv("DB_NAME", "voice_similarity").strip()
    if not user:
        raise ValueError("Chưa có DB_USER. Sao chép .env.example thành .env và điền tài khoản MySQL.")
    if not re.fullmatch(r"[A-Za-z0-9_]{1,64}", name):
        raise ValueError("DB_NAME chỉ được chứa chữ, số và dấu gạch dưới (tối đa 64 ký tự).")
    return {
        "host": os.getenv("DB_HOST", "127.0.0.1"),
        "port": int(os.getenv("DB_PORT", "3306")),
        "user": user,
        "password": os.getenv("DB_PASSWORD", ""),
        "database": name,
        "charset": "utf8mb4",
        "connection_timeout": 5,
    }


def connect(use_database=True):
    """Mở kết nối MySQL cục bộ, tùy chọn dùng database đích."""
    settings = connection_settings()
    if not use_database:
        settings.pop("database")
    return mysql.connector.connect(**settings)


def resolve_audio_path(relative_path):
    """Giải quyết đường dẫn audio tương đối và chặn truy cập ngoài thư mục dataset."""
    path = (ROOT / Path(*PureWindowsPath(relative_path).parts)).resolve()
    if not path.is_relative_to(DATA_DIR.resolve()):
        raise ValueError("Đường dẫn âm thanh nằm ngoài Dataset voices/dataset_final.")
    return path


def initialize_database(rebuild=False):
    """Tạo database và 3 bảng nếu chưa có; từ chối ghi đè dữ liệu sẵn có."""
    name = connection_settings()["database"]
    with connect(use_database=False) as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME=%s", (name,))
            exists = cursor.fetchone() is not None
            if exists:
                cursor.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA=%s", (name,))
                tables = {row[0] for row in cursor.fetchall()}
                expected = {"audio_files", "audio_features", "feature_stats"}
                if tables and (not rebuild or tables != expected):
                    raise ValueError(
                        f"CSDL {name} đã có bảng. Dùng DB_NAME mới; chỉ dùng --rebuild để ghi lại đúng 3 bảng của ứng dụng."
                    )
            else:
                cursor.execute(f"CREATE DATABASE `{name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_bin")
            cursor.execute(f"USE `{name}`")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audio_files (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    file_path VARCHAR(512) NOT NULL UNIQUE,
                    source VARCHAR(32) NOT NULL,
                    subset VARCHAR(32) NOT NULL,
                    speaker_id VARCHAR(64) NOT NULL,
                    speaker_name VARCHAR(255) NOT NULL,
                    gender VARCHAR(16) NOT NULL,
                    duration_sec DOUBLE NOT NULL,
                    sample_rate INT NOT NULL,
                    channels INT NOT NULL,
                    format VARCHAR(16) NOT NULL,
                    subtype VARCHAR(32) NOT NULL,
                    file_size_bytes BIGINT NOT NULL,
                    audio_hash CHAR(64),
                    status VARCHAR(16) NOT NULL,
                    error_message TEXT,
                    INDEX idx_speaker (source, speaker_id),
                    INDEX idx_audio_hash (audio_hash)
                ) ENGINE=InnoDB
            """)
            columns = ", ".join(f"`{name}` DOUBLE NULL" for name in FEATURE_NAMES)
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS audio_features (
                    audio_id INT PRIMARY KEY,
                    {columns},
                    FOREIGN KEY (audio_id) REFERENCES audio_files(id)
                ) ENGINE=InnoDB
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feature_stats (
                    feature_name VARCHAR(64) PRIMARY KEY,
                    median DOUBLE NOT NULL,
                    mean DOUBLE NOT NULL,
                    std DOUBLE NOT NULL
                ) ENGINE=InnoDB
            """)
            if rebuild:
                cursor.execute("DELETE FROM audio_features")
                cursor.execute("DELETE FROM audio_files")
                cursor.execute("DELETE FROM feature_stats")
        conn.commit()


def load_dataset():
    """Đọc metadata và 65 đặc trưng thô của các bản ghi ready từ MySQL."""
    columns = ", ".join(f"v.`{name}`" for name in FEATURE_NAMES)
    with connect() as conn, conn.cursor(dictionary=True) as cursor:
        cursor.execute(f"""
            SELECT a.*, {columns} FROM audio_files a
            JOIN audio_features v ON v.audio_id=a.id
            WHERE a.status='ready' ORDER BY a.file_path
        """)
        combined = cursor.fetchall()
    matrix = np.array([[row.pop(name) for name in FEATURE_NAMES] for row in combined], dtype=float)
    if not combined:
        raise ValueError("CSDL chưa có đặc trưng hợp lệ. Chạy python database.py trước.")
    return combined, matrix


def load_stats():
    """Đọc 65 dòng thống kê median/mean/std đã lưu trong feature_stats."""
    with connect() as conn, conn.cursor(dictionary=True) as cursor:
        cursor.execute("SELECT feature_name, median, mean, std FROM feature_stats")
        rows = {row["feature_name"]: row for row in cursor.fetchall()}
    if set(rows) != set(FEATURE_NAMES):
        raise ValueError("Chưa có đủ 65 tham số chuẩn hóa; quá trình xây CSDL chưa hoàn tất.")
    return {key: np.array([rows[name][key] for name in FEATURE_NAMES], dtype=float)
            for key in ("median", "mean", "std")}


def get_overview():
    """Thống kê số tệp, người nói, thời lượng theo nguồn và trạng thái."""
    with connect() as conn, conn.cursor(dictionary=True) as cursor:
        cursor.execute("""
            SELECT source, status, COUNT(*) AS files,
                   COUNT(DISTINCT speaker_id) AS speakers,
                   SUM(duration_sec) AS duration_sec
            FROM audio_files GROUP BY source, status ORDER BY source, status
        """)
        return cursor.fetchall()


def build_database(rebuild=False, limit=None):
    """Duyệt manifest, trích đặc trưng từng WAV, lưu MySQL và tính thống kê chuẩn hóa."""
    with MANIFEST.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit phải lớn hơn 0.")
        rows = rows[:limit]
    if len({row["path"] for row in rows}) != len(rows):
        raise ValueError("Manifest chứa đường dẫn trùng.")
    initialize_database(rebuild=rebuild)
    start = time.perf_counter()
    failures, warnings = [], []
    columns = ", ".join(f"`{name}`" for name in FEATURE_NAMES)
    feature_sql = f"INSERT INTO audio_features (audio_id, {columns}) VALUES ({', '.join(['%s'] * 66)})"
    file_sql = """
        INSERT INTO audio_files
        (file_path, source, subset, speaker_id, speaker_name, gender,
         duration_sec, sample_rate, channels, format, subtype, file_size_bytes,
         audio_hash, status, error_message)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """
    with connect() as conn, conn.cursor() as cursor:
        for index, row in enumerate(rows, 1):
            result, error = None, None
            try:
                if row["gender"] != "male":
                    raise ValueError("Manifest chứa bản ghi không có nhãn male.")
                result = extract_features(resolve_audio_path(row["path"]))
            except (ValueError, OSError, RuntimeError) as exc:
                error = str(exc)
                failures.append({"path": row["path"], "error": error})
            parameters = (
                row["path"], row["source"], row["subset"], row["speaker_id"],
                row["speaker_name"], row["gender"], float(row["duration_sec"]),
                int(row["sample_rate"]), int(row["channels"]), row["format"],
                row["subtype"], int(row["file_size_bytes"]),
                result["audio_hash"] if result else None, "ready" if result else "failed", error,
            )
            cursor.execute(file_sql, parameters)
            if result:
                values = [None if not np.isfinite(v) else float(v) for v in result["vector"]]
                cursor.execute(feature_sql, (cursor.lastrowid, *values))
                if result["warnings"]:
                    warnings.append({"path": row["path"], "warnings": result["warnings"]})
            conn.commit()
            if index % 50 == 0 or index == len(rows) or error:
                print(f"[{index}/{len(rows)}] ready={index - len(failures)}, failed={len(failures)}", flush=True)
    metadata, matrix = load_dataset()
    stats = fit_normalizer(matrix)
    with connect() as conn, conn.cursor() as cursor:
        cursor.executemany(
            "INSERT INTO feature_stats (feature_name, median, mean, std) VALUES (%s,%s,%s,%s)",
            [(name, float(stats["median"][i]), float(stats["mean"][i]), float(stats["std"][i]))
             for i, name in enumerate(FEATURE_NAMES)],
        )
        conn.commit()
    report = {
        "database": connection_settings()["database"], "total": len(rows),
        "ready": len(metadata), "failed": len(failures),
        "speakers": len({(r["source"], r["speaker_id"]) for r in metadata}),
        "elapsed_seconds": time.perf_counter() - start,
        "missing_per_feature": dict(zip(FEATURE_NAMES, np.isnan(matrix).sum(axis=0).tolist())),
        "settings": EXTRACTOR_SETTINGS, "failures": failures, "warnings": warnings,
        "sources": get_overview(),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "build_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("total", "ready", "failed", "speakers", "elapsed_seconds")}, indent=2))
    return report


def main():
    """CLI xây dữ liệu, kiểm tra kết nối hoặc xây lại có xác nhận."""
    parser = argparse.ArgumentParser(description="Tạo 3 bảng MySQL và trích đặc trưng bộ giọng nói.")
    parser.add_argument("--rebuild", action="store_true", help="XÓA dữ liệu dẫn xuất trong 3 bảng ứng dụng rồi xây lại.")
    parser.add_argument("--limit", type=int, help="Chỉ dùng để chạy thử trong CSDL riêng.")
    parser.add_argument("--check", action="store_true", help="Chỉ kiểm tra kết nối và liệt kê bảng, không sửa dữ liệu.")
    args = parser.parse_args()
    try:
        if args.check:
            name = connection_settings()["database"]
            with connect(use_database=False) as conn, conn.cursor() as cursor:
                cursor.execute("SELECT VERSION()")
                print("MySQL:", cursor.fetchone()[0])
                cursor.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA=%s", (name,))
                print("Database:", name, "Tables:", [row[0] for row in cursor.fetchall()])
        else:
            build_database(rebuild=args.rebuild, limit=args.limit)
    except (ValueError, OSError, mysql.connector.Error) as exc:
        parser.exit(1, f"Lỗi: {exc}\n")


if __name__ == "__main__":
    main()
