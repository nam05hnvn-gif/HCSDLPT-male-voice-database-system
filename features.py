import hashlib
import io
import math
from pathlib import Path
import struct
import warnings as python_warnings

import numpy as np
import parselmouth
from parselmouth.praat import call
from scipy.fft import dct
from scipy.signal import resample_poly
from scipy.signal.windows import hann
import soundfile as sf


FEATURE_NAMES = (
    "rms_mean", "rms_std", "zcr_mean", "zcr_std", "silence_ratio",
    "f0_mean", "f0_std", "f0_range",
    "f1_mean", "f1_std", "f2_mean", "f2_std", "f3_mean", "f3_std",
    "hnr_mean", "hnr_std", "jitter_local", "shimmer_local",
    "centroid_mean", "centroid_std", "bandwidth_mean", "bandwidth_std",
    "rolloff_mean", "rolloff_std", "flux_mean", "flux_std",
) + tuple(
    f"mfcc_{i}_{stat}" for i in range(13) for stat in ("mean", "std")
) + tuple(f"delta_{i}_std" for i in range(13))

FEATURE_GROUPS = {
    "time": (0, 5),
    "voice": (5, 18),
    "spectral": (18, 26),
    "cepstral": (26, 65),
}

FEATURE_UNITS = (
    "amplitude", "amplitude", "ratio", "ratio", "ratio",
    "Hz", "Hz", "Hz", "Hz", "Hz", "Hz", "Hz", "Hz", "Hz",
    "dB", "dB", "ratio", "ratio",
    "Hz", "Hz", "Hz", "Hz", "Hz", "Hz", "dimensionless", "dimensionless",
) + ("dimensionless",) * 26 + ("dimensionless/frame",) * 13

EXTRACTOR_SETTINGS = {
    "sample_rate": 16000,
    "max_file_size_bytes": 25 * 1024 * 1024,
    "duration_min_sec": 1.0,
    "duration_max_sec": 60.0,
    "max_original_sample_rate": 192000,
    "allowed_channels": (1, 2),
    "allowed_formats": ("WAV", "WAVEX", "FLAC"),
    "mono_conversion": "arithmetic mean of channels",
    "resampling": "scipy.signal.resample_poly",
    "resample_window": ("kaiser", 5.0),
    "resample_padtype": "constant",
    "denoise": False,
    "normalize_amplitude": False,
    "trim_silence": False,
    "remove_dc": False,
    "audio_hash": "sha256(mono16k float32 little-endian samples + uint32 little-endian sample rate)",
    "frame_length": 400,
    "hop_length": 160,
    "center": False,
    "drop_incomplete_frame": True,
    "std_ddof": 0,
    "zcr": "count(x[n] * x[n+1] < 0) / (frame_length - 1)",
    "silence_rms_threshold": 0.01,
    "silence_ratio_interpretation": "low-energy proxy, not a speech VAD",
    "window": "periodic Hann",
    "n_fft": 512,
    "spectrum": "abs(rfft(windowed frame)) ** 2",
    "spectral_weighting": "power; positive-total-power frames only",
    "rolloff_fraction": 0.85,
    "flux": "Euclidean distance of L1-normalized power; adjacent positive-power frames only",
    "n_mels": 40,
    "mel_scale": "HTK: 2595 * log10(1 + frequency / 700)",
    "mel_frequency_min": 0.0,
    "mel_frequency_max": 8000.0,
    "mel_filters": "continuous-frequency triangles, unit height, no area normalization",
    "mel_log": "natural",
    "mel_energy_floor": 1e-10,
    "dct_type": 2,
    "dct_norm": "ortho",
    "n_mfcc": 13,
    "mfcc_chronology": "all frames, including silent frames",
    "delta_width": 2,
    "delta_boundary": "repeat edge frames",
    "delta_denominator": 10,
    "pitch_method": "Praat raw autocorrelation (to_pitch_ac), not Part 1 filtered AC",
    "voice_time_step": 0.01,
    "pitch_floor": 50.0,
    "pitch_ceiling": 800.0,
    "pitch_max_candidates": 15,
    "pitch_very_accurate": False,
    "pitch_silence_threshold": 0.03,
    "pitch_voicing_threshold": 0.45,
    "pitch_octave_cost": 0.01,
    "pitch_octave_jump_cost": 0.35,
    "pitch_voiced_unvoiced_cost": 0.14,
    "minimum_voiced_duration_sec": 0.5,
    "voiced_duration": "positive finite pitch samples * voice_time_step",
    "voiced_alignment": "nearest pitch sample time; ties choose earlier sample",
    "formant_method": "Praat Burg",
    "formant_max_number": 5.0,
    "formant_ceiling": 5000.0,
    "formant_window_length": 0.025,
    "formant_pre_emphasis_from": 50.0,
    "hnr_method": "Praat cross-correlation (to_harmonicity_cc)",
    "hnr_minimum_pitch": 50.0,
    "hnr_silence_threshold": 0.1,
    "hnr_periods_per_window": 1.0,
    "hnr_unvoiced_sentinel_db": -200.0,
    "pulse_method": "Sound + existing Pitch: To PointProcess (cc)",
    "pulse_time_start": 0.0,
    "pulse_time_end": 0.0,
    "pulse_period_floor": 0.8 / 800.0,
    "pulse_period_ceiling": 1.25 / 50.0,
    "pulse_max_period_factor": 1.3,
    "pulse_max_amplitude_factor": 1.6,
    "jitter_shimmer_units": "fraction, not percent",
    "missing_values": "NaN; gallery median imputation before population z-score",
    "constant_features": "std=0 gives z=0 for gallery and query",
}


def _validate_wave_header(data):
    byte_order = "little" if data[:4] == b"RIFF" else "big"
    end = int.from_bytes(data[4:8], byte_order) + 8
    if end < 12 or end > len(data):
        raise ValueError("Header WAV khai báo kích thước không hợp lệ hoặc tệp đã bị cắt cụt.")
    offset = 12
    found = set()
    while offset < end:
        if offset + 8 > end:
            raise ValueError("Header khối dữ liệu WAV bị thiếu.")
        tag = data[offset:offset + 4]
        size = int.from_bytes(data[offset + 4:offset + 8], byte_order)
        chunk_end = offset + 8 + size
        if chunk_end > end:
            raise ValueError("Khối dữ liệu WAV vượt kích thước khai báo hoặc đã bị cắt cụt.")
        if tag in (b"fmt ", b"data"):
            found.add(tag)
        offset = chunk_end + (size % 2)
        if offset > end:
            raise ValueError("Khối dữ liệu WAV thiếu byte đệm bắt buộc.")
    if found != {b"fmt ", b"data"}:
        raise ValueError("Tệp WAV thiếu khối định dạng hoặc khối dữ liệu âm thanh.")


def read_audio(source):
    settings = EXTRACTOR_SETTINGS
    limit = settings["max_file_size_bytes"]
    if isinstance(source, bytes):
        data = source
    elif isinstance(source, (str, Path)):
        try:
            with Path(source).open("rb") as stream:
                data = stream.read(limit + 1)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Không thể đọc tệp âm thanh: {exc}") from exc
    else:
        raise ValueError("Nguồn âm thanh phải là đường dẫn (str/Path) hoặc bytes.")
    if len(data) > limit:
        raise ValueError("Tệp âm thanh vượt giới hạn 25 MiB.")
    if not data:
        raise ValueError("Tệp âm thanh rỗng.")
    is_wave = len(data) >= 12 and data[:4] in (b"RIFF", b"RIFX") and data[8:12] == b"WAVE"
    is_flac = data.startswith(b"fLaC")
    if not (is_wave or is_flac):
        raise ValueError("Nội dung tệp không phải WAV/WAVEX hoặc FLAC hợp lệ.")
    if is_wave:
        # libsndfile can silently shorten truncated WAV data to fit the available bytes.
        _validate_wave_header(data)
    try:
        with sf.SoundFile(io.BytesIO(data)) as audio:
            original_rate = audio.samplerate
            channels = audio.channels
            frame_count = audio.frames
            audio_format = audio.format
            subtype = audio.subtype
            if audio_format not in settings["allowed_formats"]:
                raise ValueError("Chỉ chấp nhận nội dung âm thanh WAV/WAVEX hoặc FLAC.")
            if (audio_format == "FLAC") != is_flac:
                raise ValueError("Định dạng giải mã không khớp với header âm thanh.")
            if not 1 <= original_rate <= settings["max_original_sample_rate"]:
                raise ValueError("Tần số lấy mẫu không hợp lệ hoặc vượt giới hạn 192000 Hz.")
            if channels not in settings["allowed_channels"]:
                raise ValueError("Âm thanh phải có 1 hoặc 2 kênh.")
            duration = frame_count / original_rate
            if not settings["duration_min_sec"] <= duration <= settings["duration_max_sec"]:
                raise ValueError("Thời lượng âm thanh phải từ 1 đến 60 giây.")
            decoded = audio.read(frames=frame_count + 1, dtype="float64", always_2d=True)
    except (sf.LibsndfileError, OSError, EOFError) as exc:
        raise ValueError(f"Không thể giải mã tệp âm thanh WAV/FLAC: {exc}") from exc
    if decoded.shape != (frame_count, channels):
        raise ValueError("Dữ liệu âm thanh bị thiếu hoặc không khớp với header.")
    if not np.isfinite(decoded).all():
        raise ValueError("Âm thanh chứa mẫu NaN hoặc vô cực, không thể xử lý.")
    with np.errstate(over="ignore", invalid="ignore"):
        y = decoded.mean(axis=1)
    if not np.isfinite(y).all():
        raise ValueError("Biên độ âm thanh quá lớn để chuyển thành mono an toàn.")
    sample_rate = settings["sample_rate"]
    if original_rate != sample_rate:
        factor = math.gcd(original_rate, sample_rate)
        with np.errstate(over="ignore", invalid="ignore"):
            y = resample_poly(
                y, sample_rate // factor, original_rate // factor,
                window=settings["resample_window"], padtype=settings["resample_padtype"],
            )
    y = np.ascontiguousarray(y, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        canonical = y.astype("<f4")
    if not np.isfinite(y).all() or not np.isfinite(canonical).all():
        raise ValueError("Âm thanh sau chuyển đổi chứa mẫu không hữu hạn hoặc biên độ quá lớn.")
    if not np.any(y != 0):
        raise ValueError("Âm thanh im lặng hoàn toàn sau khi chuyển thành mono.")
    digest = hashlib.sha256(canonical.tobytes())
    digest.update(struct.pack("<I", sample_rate))
    info = {
        "original_sample_rate": original_rate,
        "original_channels": channels,
        "duration_sec": y.size / sample_rate,
        "format": audio_format,
        "subtype": subtype,
        "file_size_bytes": len(data),
        "sample_rate": sample_rate,
        "audio_hash": digest.hexdigest(),
    }
    return y, info


def frame_signal(y, frame_length=400, hop_length=160):
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1 or not np.isfinite(y).all():
        raise ValueError("Tín hiệu chia khung phải là mảng mono một chiều hữu hạn.")
    if not isinstance(frame_length, (int, np.integer)) or frame_length < 2:
        raise ValueError("Độ dài khung phải là số nguyên ít nhất bằng 2.")
    if not isinstance(hop_length, (int, np.integer)) or hop_length < 1:
        raise ValueError("Bước khung phải là số nguyên dương.")
    if y.size < frame_length:
        return np.empty((0, frame_length), dtype=np.float64)
    return np.lib.stride_tricks.sliding_window_view(y, frame_length)[::hop_length]


def calculate_spectrum(frames, n_fft=512):
    frames = np.asarray(frames, dtype=np.float64)
    if frames.ndim != 2 or frames.shape[1] < 2 or not np.isfinite(frames).all():
        raise ValueError("Các khung phổ phải là mảng hai chiều hữu hạn.")
    if not isinstance(n_fft, (int, np.integer)) or n_fft < frames.shape[1]:
        raise ValueError("Kích thước FFT phải là số nguyên không nhỏ hơn độ dài khung.")
    window = hann(frames.shape[1], sym=False)
    return np.abs(np.fft.rfft(frames * window, n=n_fft, axis=1)) ** 2


def spectral_features(power, sample_rate=16000, n_fft=512, rolloff_fraction=0.85):
    power = np.asarray(power, dtype=np.float64)
    if power.ndim != 2 or power.shape[1] != n_fft // 2 + 1:
        raise ValueError("Ma trận công suất không khớp với kích thước FFT.")
    if not np.isfinite(power).all() or np.any(power < 0):
        raise ValueError("Công suất phải hữu hạn và không âm.")
    if sample_rate <= 0 or not 0 < rolloff_fraction <= 1:
        raise ValueError("Tần số lấy mẫu hoặc tỷ lệ rolloff không hợp lệ.")
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    totals = power.sum(axis=1)
    valid = totals > 0
    normalized = np.zeros_like(power)
    normalized[valid] = power[valid] / totals[valid, None]
    centroid = np.full(power.shape[0], np.nan)
    bandwidth = np.full(power.shape[0], np.nan)
    rolloff = np.full(power.shape[0], np.nan)
    centroid[valid] = normalized[valid] @ frequencies
    bandwidth[valid] = np.sqrt(np.sum(
        normalized[valid] * (frequencies - centroid[valid, None]) ** 2, axis=1,
    ))
    indices = np.argmax(
        np.cumsum(power[valid], axis=1) >= rolloff_fraction * totals[valid, None], axis=1,
    )
    rolloff[valid] = frequencies[indices]
    adjacent = valid[1:] & valid[:-1]
    flux = np.full(max(0, power.shape[0] - 1), np.nan)
    flux[adjacent] = np.linalg.norm(
        normalized[1:][adjacent] - normalized[:-1][adjacent], axis=1,
    )
    return {"centroid": centroid, "bandwidth": bandwidth, "rolloff": rolloff, "flux": flux}


def mel_filterbank(sample_rate=16000, n_fft=512, n_mels=40, fmin=0.0, fmax=None):
    if fmax is None:
        fmax = sample_rate / 2.0
    if sample_rate <= 0 or not 0 <= fmin < fmax <= sample_rate / 2.0:
        raise ValueError("Dải tần Mel phải nằm trong [0, sample_rate/2].")
    if not isinstance(n_mels, (int, np.integer)) or n_mels < 1:
        raise ValueError("Số bộ lọc Mel phải là số nguyên dương.")
    if not isinstance(n_fft, (int, np.integer)) or n_fft < 2:
        raise ValueError("Kích thước FFT phải là số nguyên ít nhất bằng 2.")
    mel_limits = 2595.0 * np.log10(1.0 + np.array([fmin, fmax]) / 700.0)
    edges = 700.0 * (10.0 ** (np.linspace(*mel_limits, n_mels + 2) / 2595.0) - 1.0)
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    rising = (frequencies[None, :] - edges[:-2, None]) / (edges[1:-1] - edges[:-2])[:, None]
    falling = (edges[2:, None] - frequencies[None, :]) / (edges[2:] - edges[1:-1])[:, None]
    return np.maximum(0.0, np.minimum(rising, falling))


def calculate_delta(mfcc, width=2):
    mfcc = np.asarray(mfcc, dtype=np.float64)
    if mfcc.ndim not in (1, 2) or not np.isfinite(mfcc).all():
        raise ValueError("Chuỗi tính delta phải là mảng một/hai chiều hữu hạn, trục đầu là thời gian.")
    if not isinstance(width, (int, np.integer)) or width < 1:
        raise ValueError("Độ rộng delta phải là số nguyên dương.")
    if len(mfcc) == 0:
        return np.empty_like(mfcc)
    padding = [(width, width)] + [(0, 0)] * (mfcc.ndim - 1)
    padded = np.pad(mfcc, padding, mode="edge")
    delta = np.zeros_like(mfcc)
    for offset in range(1, width + 1):
        delta += offset * (
            padded[width + offset:width + offset + len(mfcc)]
            - padded[width - offset:width - offset + len(mfcc)]
        )
    return delta / (2 * sum(offset ** 2 for offset in range(1, width + 1)))


def _mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    return float(values.mean()), float(values.std(ddof=0))


def _voiced_at_times(times, pitch_times, voiced):
    right = np.searchsorted(pitch_times, times)
    right = np.clip(right, 0, len(pitch_times) - 1)
    left = np.maximum(right - 1, 0)
    nearest = np.where(
        np.abs(times - pitch_times[left]) <= np.abs(times - pitch_times[right]), left, right,
    )
    return voiced[nearest]


def _voice_features(y, include_frames):
    s = EXTRACTOR_SETTINGS
    notices = []
    sound = parselmouth.Sound(y, sampling_frequency=s["sample_rate"])
    try:
        pitch = sound.to_pitch_ac(
            time_step=s["voice_time_step"], pitch_floor=s["pitch_floor"],
            pitch_ceiling=s["pitch_ceiling"], max_number_of_candidates=s["pitch_max_candidates"],
            very_accurate=s["pitch_very_accurate"], silence_threshold=s["pitch_silence_threshold"],
            voicing_threshold=s["pitch_voicing_threshold"], octave_cost=s["pitch_octave_cost"],
            octave_jump_cost=s["pitch_octave_jump_cost"],
            voiced_unvoiced_cost=s["pitch_voiced_unvoiced_cost"],
        )
    except parselmouth.PraatError as exc:
        raise ValueError(f"Praat không thể đo cao độ của âm thanh: {exc}") from exc
    f0 = pitch.selected_array["frequency"].copy()
    pitch_times = pitch.xs()
    voiced = np.isfinite(f0) & (f0 > 0)
    voiced_duration = float(np.count_nonzero(voiced) * s["voice_time_step"])
    if voiced_duration < s["minimum_voiced_duration_sec"]:
        raise ValueError(
            f"Âm thanh chỉ có {voiced_duration:.2f} giây hữu thanh hợp lệ; cần ít nhất 0,5 giây."
        )
    f0[~voiced] = np.nan
    values = [*_mean_std(f0), float(np.ptp(f0[voiced]))]
    voice_frames = {"pitch_time": pitch_times, "f0": f0, "voiced_mask": voiced} if include_frames else {}
    formant_values = np.empty((0, 3), dtype=np.float64)
    try:
        formant = sound.to_formant_burg(
            time_step=s["voice_time_step"], max_number_of_formants=s["formant_max_number"],
            maximum_formant=s["formant_ceiling"], window_length=s["formant_window_length"],
            pre_emphasis_from=s["formant_pre_emphasis_from"],
        )
        times = formant.xs()
        mask = _voiced_at_times(times, pitch_times, voiced)
        formant_values = np.full((len(times), 3), np.nan)
        for index in np.flatnonzero(mask):
            for number in range(1, 4):
                value = formant.get_value_at_time(number, float(times[index]))
                if np.isfinite(value) and value > 0:
                    formant_values[index, number - 1] = value
        if include_frames:
            voice_frames.update(formant_time=times, formants=formant_values)
    except parselmouth.PraatError as exc:
        formant_values = np.empty((0, 3), dtype=np.float64)
        notices.append(f"Không đo được formant Burg: {exc}")
    for number in range(3):
        values.extend(_mean_std(formant_values[:, number]))
    hnr = np.empty(0, dtype=np.float64)
    try:
        harmonicity = sound.to_harmonicity_cc(
            time_step=s["voice_time_step"], minimum_pitch=s["hnr_minimum_pitch"],
            silence_threshold=s["hnr_silence_threshold"],
            periods_per_window=s["hnr_periods_per_window"],
        )
        times = harmonicity.xs()
        hnr = harmonicity.values[0].copy()
        valid = _voiced_at_times(times, pitch_times, voiced) & np.isfinite(hnr)
        valid &= hnr != s["hnr_unvoiced_sentinel_db"]
        hnr[~valid] = np.nan
        if include_frames:
            voice_frames.update(hnr_time=times, hnr=hnr)
    except parselmouth.PraatError as exc:
        hnr = np.empty(0, dtype=np.float64)
        notices.append(f"Không đo được HNR cross-correlation: {exc}")
    values.extend(_mean_std(hnr))
    jitter = shimmer = np.nan
    pulse_args = (
        s["pulse_time_start"], s["pulse_time_end"], s["pulse_period_floor"],
        s["pulse_period_ceiling"], s["pulse_max_period_factor"],
    )
    try:
        pulses = call([sound, pitch], "To PointProcess (cc)")
    except parselmouth.PraatError as exc:
        notices.append(f"Không đo được chuỗi xung Praat: {exc}")
    else:
        try:
            jitter = float(call(pulses, "Get jitter (local)", *pulse_args))
        except parselmouth.PraatError as exc:
            notices.append(f"Không đo được jitter local: {exc}")
        try:
            shimmer = float(call(
                [sound, pulses], "Get shimmer (local)", *pulse_args, s["pulse_max_amplitude_factor"],
            ))
        except parselmouth.PraatError as exc:
            notices.append(f"Không đo được shimmer local: {exc}")
    values.extend(value if np.isfinite(value) and value >= 0 else np.nan for value in (jitter, shimmer))
    return values, voiced_duration, notices, voice_frames


def extract_features(source, include_frames=False):
    y, info = read_audio(source)
    s = EXTRACTOR_SETTINGS
    notices = []
    if info["duration_sec"] < 3.0:
        notices.append("Âm thanh ngắn hơn 3 giây; nên dùng bản ghi từ khoảng 6 giây để so sánh ổn định hơn.")
    frames = frame_signal(y, s["frame_length"], s["hop_length"])
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    zcr = np.count_nonzero(frames[:, :-1] * frames[:, 1:] < 0, axis=1) / (frames.shape[1] - 1)
    values = [*_mean_std(rms), *_mean_std(zcr), float(np.mean(rms <= s["silence_rms_threshold"]))]
    with python_warnings.catch_warnings(record=True) as captured:
        python_warnings.simplefilter("always", parselmouth.PraatWarning)
        voice_values, voiced_duration, voice_notices, voice_frames = _voice_features(y, include_frames)
    notices.extend(f"Cảnh báo Praat: {warning.message}" for warning in captured)
    notices.extend(voice_notices)
    values.extend(voice_values)
    power = calculate_spectrum(frames, s["n_fft"])
    spectral = spectral_features(power, s["sample_rate"], s["n_fft"], s["rolloff_fraction"])
    for name in ("centroid", "bandwidth", "rolloff", "flux"):
        values.extend(_mean_std(spectral[name]))
    filters = mel_filterbank(
        s["sample_rate"], s["n_fft"], s["n_mels"], s["mel_frequency_min"], s["mel_frequency_max"],
    )
    log_energy = np.log(np.maximum(power @ filters.T, s["mel_energy_floor"]))
    mfcc = dct(log_energy, type=s["dct_type"], norm=s["dct_norm"], axis=1)[:, :s["n_mfcc"]]
    delta = calculate_delta(mfcc, s["delta_width"])
    for number in range(s["n_mfcc"]):
        values.extend(_mean_std(mfcc[:, number]))
    values.extend(delta.std(axis=0, ddof=0))
    vector = np.asarray(values, dtype=np.float64)
    missing = [name for name, value in zip(FEATURE_NAMES, vector) if np.isnan(value)]
    if missing:
        notices.append("Không đo được đặc trưng (giữ NaN, không thay bằng 0): " + ", ".join(missing))
    result = dict(info, vector=vector, voiced_duration_sec=voiced_duration, warnings=notices)
    if include_frames:
        times = (np.arange(len(frames)) * s["hop_length"] + s["frame_length"] / 2) / s["sample_rate"]
        result["frames"] = dict(
            time=times, rms=rms, zcr=zcr, power=power, mfcc=mfcc, delta=delta,
            frequency=np.fft.rfftfreq(s["n_fft"], d=1.0 / s["sample_rate"]),
            flux_time=times[1:], **spectral, **voice_frames,
        )
    return result


def _feature_matrix(matrix, allow_vector):
    try:
        data = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Đặc trưng phải là mảng số; chỉ NaN được dùng cho giá trị thiếu.") from exc
    if data.ndim == 1 and allow_vector:
        if data.shape != (len(FEATURE_NAMES),):
            raise ValueError("Vector đặc trưng phải có đúng 65 phần tử.")
    elif data.ndim != 2 or data.shape[1] != len(FEATURE_NAMES):
        raise ValueError("Ma trận đặc trưng phải có dạng (số mẫu, 65).")
    if np.isinf(data).any():
        raise ValueError("Đặc trưng chứa vô cực; chỉ NaN được dùng cho giá trị thiếu.")
    return data


def fit_normalizer(matrix):
    data = _feature_matrix(matrix, allow_vector=False)
    if data.shape[0] == 0:
        raise ValueError("Gallery rỗng, không thể tính thống kê chuẩn hóa.")
    all_missing = np.isnan(data).all(axis=0)
    if all_missing.any():
        names = ", ".join(np.asarray(FEATURE_NAMES)[all_missing])
        raise ValueError("Toàn bộ gallery thiếu các đặc trưng: " + names)
    with np.errstate(over="ignore", invalid="ignore"):
        median = np.nanmedian(data, axis=0)
        imputed = np.where(np.isnan(data), median, data)
        mean = imputed.mean(axis=0)
        std = imputed.std(axis=0, ddof=0)
    constant = np.all(imputed == imputed[0], axis=0)
    mean[constant] = imputed[0, constant]
    std[constant] = 0.0
    if not all(np.isfinite(values).all() for values in (median, mean, std)):
        raise ValueError("Đặc trưng quá lớn để tính thống kê chuẩn hóa hữu hạn.")
    return {"median": median, "mean": mean, "std": std}


def normalize(matrix, stats):
    data = _feature_matrix(matrix, allow_vector=True)
    parameters = {}
    for name in ("median", "mean", "std"):
        try:
            values = np.asarray(stats[name], dtype=np.float64)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Thiếu hoặc sai thống kê chuẩn hóa '{name}'.") from exc
        if values.shape != (len(FEATURE_NAMES),) or not np.isfinite(values).all():
            raise ValueError(f"Thống kê '{name}' phải chứa đúng 65 số hữu hạn.")
        parameters[name] = values
    if np.any(parameters["std"] < 0):
        raise ValueError("Độ lệch chuẩn đã lưu không được âm.")
    imputed = np.where(np.isnan(data), parameters["median"], data)
    active = parameters["std"] > 0
    result = np.zeros_like(imputed)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result[..., active] = (
            imputed[..., active] - parameters["mean"][active]
        ) / parameters["std"][active]
    if not np.isfinite(result).all():
        raise ValueError("Kết quả chuẩn hóa vượt giới hạn số hữu hạn.")
    return result
