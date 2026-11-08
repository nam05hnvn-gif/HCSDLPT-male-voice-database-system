"""Giao diện Streamlit cho demo tìm kiếm giọng nói bằng 65 đặc trưng."""

import hashlib
import json
import math
from pathlib import Path, PureWindowsPath

import mysql.connector
import numpy as np
import pandas as pd
import streamlit as st

from database import get_overview, resolve_audio_path
from features import read_audio
from search import search_audio


MAX_UPLOAD_BYTES = 25 * 1024 * 1024
GROUPS = (
    ("time", "Thời gian", 5),
    ("voice", "Giọng nói", 13),
    ("spectral", "Phổ", 8),
    ("cepstral", "Cepstral", 39),
)
GROUP_LABELS = {key: label for key, label, _ in GROUPS}
DB_GUIDANCE = (
    "Kiểm tra dịch vụ MySQL, cấu hình tài khoản trong `.env` "
    "(tham khảo `.env.example`), rồi chạy `python database.py` để chuẩn bị dữ liệu."
)


def show_database_overview():
    with st.sidebar:
        st.header("Dữ liệu MySQL")
        try:
            overview = get_overview()
        except (ValueError, mysql.connector.Error, OSError):
            st.error("Chưa đọc được cơ sở dữ liệu.")
            st.info(DB_GUIDANCE)
        else:
            if not overview:
                st.warning("Cơ sở dữ liệu chưa có bản ghi.")
                st.info(DB_GUIDANCE)
            else:
                ready = [row for row in overview if row["status"] == "ready"]
                failed = sum(int(row["files"]) for row in overview if row["status"] == "failed")
                st.metric("Tổng số tệp", sum(int(row["files"]) for row in overview))
                st.metric("Người nói sẵn sàng", sum(int(row["speakers"]) for row in ready))
                st.metric("Nguồn dữ liệu", len({row["source"] for row in overview}))
                st.caption("Người nói được đếm riêng theo từng nguồn, chỉ tính trạng thái ready.")
                st.dataframe(
                    pd.DataFrame([
                        {
                            "Nguồn": row["source"],
                            "Trạng thái": row["status"],
                            "Tệp": int(row["files"]),
                            "Người nói": int(row["speakers"]),
                            "Phút": round(float(row["duration_sec"] or 0) / 60, 2),
                        }
                        for row in overview
                    ]),
                    hide_index=True,
                    width="stretch",
                )
                st.caption("Số người nói trong bảng là số riêng của mỗi nhóm nguồn / trạng thái.")
                if failed:
                    st.warning(f"Có {failed} tệp lỗi (failed), không dùng để so sánh.")
                if not ready:
                    st.warning("Chưa có tệp sẵn sàng để tìm kiếm.")
                    st.info(DB_GUIDANCE)
                else:
                    st.success("Đã đọc được tổng quan dữ liệu.")
        st.divider()
        st.subheader("Phương pháp")
        st.write("65 đặc trưng: 5 thời gian + 13 giọng nói + 8 phổ + 39 cepstral.")
        st.caption(
            "Giá trị thiếu được điền bằng trung vị của bộ dữ liệu; chuẩn hóa z-score "
            "với thống kê đã lưu, sau đó so sánh khoảng cách Euclid. "
            "Giao diện không huấn luyện lại tham số chuẩn hóa."
        )
        st.caption("3 bảng MySQL: audio_files, audio_features, feature_stats.")


def clear_upload_state(upload_hash):
    """Không dùng query_hash của bộ trích xuất thay cho hash của byte tải lên."""
    if st.session_state.get("voice_upload_hash") != upload_hash:
        for key in ("voice_result", "voice_preview", "voice_error", "voice_detail"):
            st.session_state.pop(key, None)
        st.session_state["voice_upload_hash"] = upload_hash


def prepare_preview(audio_bytes):
    """Chỉ đọc / tiền xử lý âm thanh, không trích pitch hoặc bộ 65 đặc trưng."""
    if "voice_preview" not in st.session_state:
        try:
            waveform, info = read_audio(audio_bytes)
            if len(waveform) == 0:
                raise ValueError("Empty audio")
            positions = np.linspace(0, len(waveform) - 1, min(2000, len(waveform)), dtype=int)
            chart = pd.DataFrame({
                "Thời gian (giây)": positions / 16000.0,
                "Biên độ": waveform[positions],
            })
            st.session_state["voice_preview"] = {"chart": chart, "info": info}
        except (ValueError, OSError):
            st.session_state["voice_preview"] = None
    return st.session_state["voice_preview"]


def show_warnings(payload):
    warnings = list(payload["query_info"].get("warnings", []))
    warnings.extend(payload.get("warnings", []))
    for message in dict.fromkeys(warnings):
        st.warning(str(message))


def make_top_table(results):
    return pd.DataFrame([
        {
            "Hạng": item["rank"],
            "Tệp": PureWindowsPath(item["file_path"]).name,
            "Người nói": item.get("speaker_name") or item["speaker_id"],
            "ID người nói": item["speaker_id"],
            "Nguồn": item["source"],
            "Thời lượng (giây)": item["duration_sec"],
            "Khoảng cách d": item["distance"],
            "Điểm s": item["similarity"],
        }
        for item in results
    ])


def show_players(results):
    for item in results:
        filename = PureWindowsPath(item["file_path"]).name
        speaker = item.get("speaker_name") or item["speaker_id"]
        with st.expander(f"#{item['rank']} · {filename} · {speaker} / {item['source']}"):
            st.caption(
                f"ID: {item['speaker_id']} · {item['duration_sec']:.2f} giây · "
                f"d = {item['distance']:.6f} · s = {item['similarity']:.6f}"
            )
            for warning in item.get("warnings", []):
                st.warning(str(warning))
            try:
                safe_path = Path(resolve_audio_path(item["file_path"]))
                audio_bytes = safe_path.read_bytes()
            except (ValueError, OSError, mysql.connector.Error):
                st.warning("Không phát được tệp này. Kiểm tra tệp và quyền đọc trong thư mục dataset.")
            else:
                audio_format = "audio/flac" if safe_path.suffix.lower() == ".flac" else "audio/wav"
                st.audio(audio_bytes, format=audio_format)


def show_calculations(payload, results):
    with st.expander("Các bước tính toán"):
        st.markdown("**1. Tiền xử lý và trích đặc trưng**")
        st.write(
            "Âm thanh được đưa về mono 16 kHz và trích đúng 65 đặc trưng: "
            "5 thời gian, 13 giọng nói, 8 phổ, 39 cepstral. "
            "Bảng dưới lấy trực tiếp kết quả của pipeline tìm kiếm."
        )
        st.markdown("**2. Điền giá trị thiếu và chuẩn hóa z-score**")
        st.latex(r"z_j = \frac{x_j^{*} - \mu_j}{\sigma_j}")
        st.caption(
            "raw: giá trị gốc (ô trống là thiếu); imputed: giá trị sau điền trung vị; "
            "z: giá trị đã chuẩn hóa. Trung vị, trung bình và độ lệch chuẩn lấy từ "
            "feature_stats của bộ dữ liệu, không tính lại trên tệp truy vấn."
        )
        features = pd.DataFrame([
            {
                "Đặc trưng": row["feature"],
                "Nhóm": GROUP_LABELS.get(row["group"], row["group"]),
                "Đơn vị": row["unit"],
                "raw": row["raw"],
                "imputed": row["imputed"],
                "z": row["z"],
            }
            for row in payload["query_features"]
        ])
        example = payload["query_features"][0]
        mean = payload["normalizer"]["mean"][0]
        std = payload["normalizer"]["std"][0]
        if std == 0:
            st.caption(f"Ví dụ {example['feature']}: độ lệch chuẩn bằng 0 nên z = 0.")
        else:
            st.caption(
                f"Ví dụ {example['feature']}: z = ({example['imputed']:.6f} − "
                f"{mean:.6f}) / {std:.6f} = {example['z']:.6f}."
            )
        st.caption(f"{len(features)} / 65 đặc trưng của truy vấn")
        st.dataframe(
            features,
            hide_index=True,
            width="stretch",
            column_config={name: st.column_config.NumberColumn(format="%.6f") for name in ("raw", "imputed", "z")},
        )
        st.markdown("**3. Khoảng cách Euclid và điểm tương đồng**")
        st.latex(r"d(q,c) = \sqrt{\sum_{j=1}^{65}(z_{q,j}-z_{c,j})^2}, \qquad s = \frac{1}{1+d}")
        st.write("Sắp xếp d tăng dần (tương đương s giảm dần), lấy tối đa 5 tệp.")
        st.caption("Đóng góp mỗi nhóm là tổng bình phương sai khác, không phải phần trăm hoặc khoảng cách riêng.")
        st.latex(r"d^2 = C_{time} + C_{voice} + C_{spectral} + C_{cepstral}")
        if not results:
            return
        contributions = []
        for item in results:
            groups = item["group_contributions"]
            row = {"Hạng": item["rank"]}
            row.update({f"{label} ({count})": groups[key] for key, label, count in GROUPS})
            row["Tổng đóng góp"] = sum(groups[key] for key, _, _ in GROUPS)
            row["d²"] = item["distance"] ** 2
            contributions.append(row)
        group_table = pd.DataFrame(contributions)
        st.dataframe(
            group_table,
            hide_index=True,
            width="stretch",
            column_config={name: st.column_config.NumberColumn(format="%.6f") for name in group_table.columns if name != "Hạng"},
        )
        selected = st.selectbox(
            "Xem 65 sai khác của kết quả",
            options=list(range(len(results))),
            format_func=lambda index: f"#{results[index]['rank']} · {PureWindowsPath(results[index]['file_path']).name}",
            key="voice_detail",
        )
        item = results[selected]
        differences = pd.DataFrame(item["feature_differences"])
        st.dataframe(
            differences.rename(columns={
                "feature": "Đặc trưng",
                "query_z": "z truy vấn",
                "candidate_z": "z kết quả",
                "difference": "Sai khác",
                "squared_difference": "Bình phương sai khác",
            }),
            hide_index=True,
            width="stretch",
        )
        squared_sum = sum(row["squared_difference"] for row in item["feature_differences"])
        group_sum = sum(item["group_contributions"][key] for key, _, _ in GROUPS)
        distance_squared = item["distance"] ** 2
        st.caption(
            f"Tổng 65 bình phương sai khác = {squared_sum:.6f}; "
            f"tổng 4 nhóm = {group_sum:.6f}; d² = {distance_squared:.6f}. "
            "Các giá trị hiển thị được làm tròn."
        )
        if not all(math.isclose(value, distance_squared, rel_tol=1e-8, abs_tol=1e-8) for value in (squared_sum, group_sum)):
            st.warning("Tổng đóng góp chưa khớp d²; cần kiểm tra dữ liệu kết quả của pipeline.")


def show_results(payload):
    st.divider()
    st.header("Kết quả tương đồng")
    info = payload["query_info"]
    columns = st.columns(4)
    columns[0].metric("Thời lượng truy vấn", f"{info['duration_sec']:.2f} s")
    columns[1].metric("Tần số lấy mẫu gốc", f"{info['original_sample_rate']:,} Hz")
    columns[2].metric("Số kênh gốc", info["original_channels"])
    columns[3].metric("Thời lượng hữu thanh", f"{info['voiced_duration_sec']:.2f} s")
    if info["duration_sec"] < 3:
        st.warning("Truy vấn ngắn hơn 3 giây; nên dùng đoạn nói dài hơn để so sánh ổn định hơn.")
    show_warnings(payload)
    timings = payload["timings"]
    columns = st.columns(4)
    for column, label, key in zip(
        columns,
        ("Nạp dữ liệu", "Trích đặc trưng", "So sánh", "Tổng thời gian"),
        ("load_seconds", "extraction_seconds", "comparison_seconds", "total_seconds"),
    ):
        column.metric(label, f"{timings[key]:.3f} s")
    st.caption(
        f"Số ứng viên: {payload['candidates_count']} · "
        f"Số bản ghi trùng âm thanh đã loại: {payload['excluded_self_count']}. "
        "Thời gian do pipeline báo cáo, không bao gồm vẽ giao diện và phát âm thanh."
    )
    results = payload["results"][:5]
    st.subheader(f"Top {len(results)} tệp gần nhất")
    st.info(
        "Điểm s = 1 / (1 + d): càng cao càng gần về đặc trưng. "
        "Đây KHÔNG phải xác suất, phần trăm tương đồng hay độ chính xác nhận dạng. "
        "Cả 5 kết quả có thể cùng một người nói."
    )
    if results:
        top_table = make_top_table(results)
        st.dataframe(
            top_table,
            hide_index=True,
            width="stretch",
            column_config={
                "Khoảng cách d": st.column_config.NumberColumn(format="%.6f"),
                "Điểm s": st.column_config.NumberColumn(format="%.6f"),
                "Thời lượng (giây)": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        show_players(results)
    else:
        top_table = pd.DataFrame(columns=["Hạng", "Tệp", "Người nói", "ID người nói", "Nguồn", "Thời lượng (giây)", "Khoảng cách d", "Điểm s"])
        st.warning("Không còn ứng viên phù hợp sau khi loại bản ghi trùng âm thanh.")
    show_calculations(payload, results)
    left, right = st.columns(2)
    left.download_button(
        "Tải kết quả JSON",
        data=json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"),
        file_name="ket_qua_giong_noi.json",
        mime="application/json",
    )
    right.download_button(
        "Tải Top 5 CSV (Excel)",
        data=top_table.to_csv(index=False, float_format="%.6f").encode("utf-8-sig"),
        file_name="top5_giong_noi.csv",
        mime="text/csv; charset=utf-8",
    )


def main():
    st.set_page_config(page_title="Tìm kiếm giọng nói nam", layout="wide")
    st.title("Tìm kiếm giọng nói nam")
    st.write("Demo sinh viên: tìm các tệp gần nhất về đặc trưng âm học trong bộ dữ liệu giọng nói nam.")
    st.caption("Input → Tiền xử lý → Trích đặc trưng → So sánh → Tương đồng → Top 5")
    st.caption("Không tự xác định giới tính và không xác thực danh tính người nói.")
    show_database_overview()

    st.subheader("Tải âm thanh truy vấn")
    upload = st.file_uploader("Chọn tệp WAV hoặc FLAC", type=["wav", "flac"], accept_multiple_files=False, max_upload_size=25)
    st.caption("WAV / FLAC, 1–60 giây, tối đa 25 MiB. Tệp tải lên chỉ được giữ trong bộ nhớ, không ghi vào dataset hoặc MySQL.")
    if upload is None:
        clear_upload_state(None)
        st.info("Chọn một đoạn giọng nói, nghe thử, rồi bấm nút tìm kiếm.")
        return
    if upload.size > MAX_UPLOAD_BYTES or upload.size == 0:
        clear_upload_state(None)
        st.error("Tệp phải có dữ liệu và không vượt quá 25 MiB. Hãy chọn hoặc cắt một đoạn ngắn hơn.")
        return
    if PureWindowsPath(upload.name).suffix.lower() not in (".wav", ".flac"):
        clear_upload_state(None)
        st.error("Chỉ hỗ trợ tệp WAV và FLAC.")
        return

    audio_bytes = upload.getvalue()
    upload_hash = hashlib.sha256(audio_bytes).hexdigest()
    clear_upload_state(upload_hash)
    preview = prepare_preview(audio_bytes)
    if preview is not None:
        st.audio(audio_bytes, format="audio/flac" if upload.name.lower().endswith(".flac") else "audio/wav")
        st.line_chart(preview["chart"], x="Thời gian (giây)", y="Biên độ", height=180)
        st.caption("Dạng sóng mono 16 kHz, lấy mẫu thưa tối đa 2.000 điểm để hiển thị.")
        if preview["info"].get("duration_sec", 0) < 3 and "voice_result" not in st.session_state:
            st.warning("Đoạn nói ngắn hơn 3 giây; kết quả có thể kém ổn định.")
    else:
        st.error("Không đọc được âm thanh hợp lệ. Hãy kiểm tra hoặc xuất lại tệp WAV / FLAC có tiếng nói.")

    if st.button("Tìm 5 giọng nói tương đồng", type="primary", disabled=preview is None):
        st.session_state.pop("voice_result", None)
        st.session_state.pop("voice_error", None)
        st.session_state.pop("voice_detail", None)
        try:
            with st.spinner("Đang trích đặc trưng và so sánh với bộ dữ liệu…"):
                payload = search_audio(audio_bytes, k=5)
                json.dumps(payload, ensure_ascii=False, allow_nan=False)
            st.session_state["voice_result"] = {"upload_hash": upload_hash, "payload": payload}
        except mysql.connector.Error:
            st.session_state["voice_error"] = "Không truy cập được MySQL. " + DB_GUIDANCE
        except ValueError:
            st.session_state["voice_error"] = (
                "Chưa thể tìm kiếm: âm thanh có thể không đủ tiếng nói hoặc dữ liệu đặc trưng chưa sẵn sàng. "
                "Thử đoạn WAV / FLAC rõ tiếng, dài ít nhất 3 giây. " + DB_GUIDANCE
            )
        except OSError:
            st.session_state["voice_error"] = (
                "Không đọc được âm thanh hoặc dữ liệu cần thiết. Kiểm tra tệp, quyền đọc và thư mục dataset. "
                + DB_GUIDANCE
            )
    if st.session_state.get("voice_error"):
        st.error(st.session_state["voice_error"])
    stored = st.session_state.get("voice_result")
    if stored is not None and stored["upload_hash"] == upload_hash:
        show_results(stored["payload"])


if __name__ == "__main__":
    main()
