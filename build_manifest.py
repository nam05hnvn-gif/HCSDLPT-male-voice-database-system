"""
build_manifest.py

Tạo lại dataset_manifest.csv cho "Dataset voices/dataset_final".
Chỉ giữ các cột có giá trị đầy đủ cho TẤT CẢ các dòng:
- path, source, subset, speaker_id, speaker_name, gender (từ metadata 2 nguồn)
- duration_sec, sample_rate, channels, format, subtype, file_size_bytes (từ file thật)

Các cột chỉ đúng cho 1 nguồn (transcript, chapter_id, video_id, utterance_id,
nationality) hoặc không cần thiết (audio_id, original_path) đã bị loại bỏ.
"""

import os
import pandas as pd
import soundfile as sf

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "Dataset voices", "dataset_final")
OUT_CSV = os.path.join(ROOT, "Dataset voices", "dataset_manifest.csv")

OLD_DIR = os.path.join(ROOT, "old datasets")
LIBRI_META = os.path.join(OLD_DIR, "dataset_final", "metadata.csv")
VOX_META = os.path.join(OLD_DIR, "vox1_meta.csv")

COLUMNS = [
    "path", "source", "subset", "speaker_id", "speaker_name", "gender",
    "duration_sec", "sample_rate", "channels", "format", "subtype",
    "file_size_bytes",
]


def norm_gender(value):
    value = str(value).strip().lower()
    return "male" if value in ("m", "male") else value


def norm_subset(value):
    return str(value).strip().replace("-clean", "")


def main():
    libri = pd.read_csv(LIBRI_META, encoding="utf-8-sig")
    libri_by_id = libri.set_index("audio_id").to_dict("index")

    vox = pd.read_csv(VOX_META, sep=None, engine="python")
    vox.columns = [c.strip() for c in vox.columns]
    vox_id_col = next(c for c in vox.columns if "VoxCeleb1" in c)
    vox_by_id = vox.set_index(vox_id_col).to_dict("index")

    records = []
    missing_lookup = 0

    for fn in sorted(os.listdir(DATA_DIR)):
        if not fn.lower().endswith(".wav"):
            continue

        path = os.path.join(DATA_DIR, fn)
        stem = os.path.splitext(fn)[0]
        info = sf.info(path)

        rec = {
            "path": f"Dataset voices\\dataset_final\\{fn}",
            "duration_sec": round(info.frames / info.samplerate, 3),
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "format": info.format,
            "subtype": info.subtype,
            "file_size_bytes": os.path.getsize(path),
        }

        if stem.startswith("id") and "_" in stem:
            # VoxCeleb: id<speaker>_...
            speaker_id = stem.split("_")[0]
            m = vox_by_id.get(speaker_id)
            if m is None:
                missing_lookup += 1
                print(f"CANH BAO: khong tim thay {speaker_id} trong vox1_meta.csv")
                m = {}
            rec.update({
                "source": "voxcelebrity",
                "subset": norm_subset(m.get("Set", "")),
                "speaker_id": speaker_id,
                "speaker_name": str(m.get("VGGFace1 ID", "")).replace("_", " "),
                "gender": norm_gender(m.get("Gender", "")),
            })
        else:
            # LibriSpeech: <speaker>-<chapter>-<utterance>
            m = libri_by_id.get(stem)
            if m is None:
                missing_lookup += 1
                print(f"CANH BAO: khong tim thay {stem} trong metadata.csv")
                m = {}
            rec.update({
                "source": "librispeech",
                "subset": norm_subset(m.get("subset", "")),
                "speaker_id": str(m.get("speaker_id", stem.split("-")[0])),
                "speaker_name": str(m.get("speaker_name", "")),
                "gender": norm_gender(m.get("gender", "")),
            })

        records.append(rec)

    df = pd.DataFrame(records, columns=COLUMNS)
    df.to_csv(OUT_CSV, index=False)

    print(f"\nDa ghi {len(df)} dong, {len(COLUMNS)} cot -> {OUT_CSV}")
    print(f"Lookup loi: {missing_lookup}")
    for src in df["source"].unique():
        sub = df[df["source"] == src]
        print(f"  {src}: {len(sub)} file, {sub['speaker_id'].nunique()} speaker, "
              f"subset={sorted(sub['subset'].unique())}")
    empty_cols = {c: int((df[c].astype(str).str.strip() == "").sum()) for c in COLUMNS}
    bad = {c: n for c, n in empty_cols.items() if n}
    print("Cot co gia tri thieu:", bad if bad else "khong co (tat ca cot deu day du)")


if __name__ == "__main__":
    main()
