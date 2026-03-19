"""
bpm_engine.py — NovaPulse AI Signal Processing Engine
rPPG-based heart rate, stress, fatigue estimation from webcam feed.
Uses MediaPipe Tasks API (FaceLandmarker) for face mesh detection.
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from scipy.signal import butter, filtfilt, find_peaks, detrend
from collections import deque

# ---------------------------------------------------------------------------
# MediaPipe FaceLandmarker — Tasks API with VIDEO mode for temporal tracking
# ---------------------------------------------------------------------------
_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets", "face_landmarker.task"
)

_face_landmarker = None
_timestamp_ms = 0


def _get_landmarker():
    """Lazily initialise the FaceLandmarker (singleton) in VIDEO mode."""
    global _face_landmarker
    if _face_landmarker is None:
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        _face_landmarker = vision.FaceLandmarker.create_from_options(options)
    return _face_landmarker


def detect_face_landmarks(rgb_frame, timestamp_ms):
    """
    Run FaceLandmarker on an RGB numpy frame in VIDEO mode.
    timestamp_ms must be monotonically increasing.
    Returns a list of NormalizedLandmark (478 points) or None.
    """
    try:
        landmarker = _get_landmarker()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        if result.face_landmarks and len(result.face_landmarks) > 0:
            return result.face_landmarks[0]
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Eye landmark indices for EAR (Eye Aspect Ratio) calculation
# ---------------------------------------------------------------------------
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
EAR_THRESHOLD = 0.22  # tuned lower for better blink sensitivity

# Forehead landmark indices for rPPG ROI — using cheek/forehead for better signal
_FOREHEAD_IDX = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288]

# Cheek ROI for secondary green channel (more stable signal)
_LEFT_CHEEK_IDX = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148]
_RIGHT_CHEEK_IDX = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377]


# ---------------------------------------------------------------------------
# 1. Forehead ROI extraction  (returns green-channel mean)
# ---------------------------------------------------------------------------
def get_forehead_roi(frame, landmarks, w, h):
    """
    Extract green-channel mean from forehead + cheeks for stronger rPPG signal.
    Returns (green_mean, roi_bbox) or (None, None).
    """
    try:
        # Forehead ROI
        xs = [int(landmarks[i].x * w) for i in _FOREHEAD_IDX]
        ys = [int(landmarks[i].y * h) for i in _FOREHEAD_IDX]

        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        # 15% padding, clamped
        pad_x = int((x2 - x1) * 0.15)
        pad_y = int((y2 - y1) * 0.15)
        x1 = max(0, x1 - pad_x)
        x2 = min(w, x2 + pad_x)
        y1 = max(0, y1 - pad_y)
        y2 = min(h, y2 + pad_y)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None, None

        # Also grab cheek ROIs for a combined, more reliable signal
        green_means = [float(np.mean(roi[:, :, 1]))]

        for cheek_idx in (_LEFT_CHEEK_IDX, _RIGHT_CHEEK_IDX):
            try:
                cxs = [int(landmarks[i].x * w) for i in cheek_idx]
                cys = [int(landmarks[i].y * h) for i in cheek_idx]
                cx1, cx2 = max(0, min(cxs)), min(w, max(cxs))
                cy1, cy2 = max(0, min(cys)), min(h, max(cys))
                croi = frame[cy1:cy2, cx1:cx2]
                if croi.size > 0:
                    green_means.append(float(np.mean(croi[:, :, 1])))
            except Exception:
                pass

        # Average of forehead + cheeks for more stable signal
        combined = float(np.mean(green_means))
        return combined, (x1, y1, x2, y2)

    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# 2. Bandpass filter  (Butterworth)
# ---------------------------------------------------------------------------
def bandpass_filter(signal, fps, lo=0.75, hi=3.5, order=3):
    """Butterworth bandpass for rPPG — BPM range 45–210."""
    try:
        nyq = fps / 2.0
        if lo / nyq >= 1.0 or hi / nyq >= 1.0:
            return signal
        b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
        return filtfilt(b, a, signal)
    except Exception:
        return signal


# ---------------------------------------------------------------------------
# 3. BPM calculation  (with detrending for accuracy)
# ---------------------------------------------------------------------------
def calc_bpm(green_buffer, fps):
    """Estimate heart rate (BPM) from the green-channel buffer."""
    min_samples = int(fps * 3)  # 3 seconds minimum (faster initial reading)
    if len(green_buffer) < min_samples:
        return None

    sig = np.array(green_buffer, dtype=np.float64)

    # Detrend to remove slow drift from head movement
    sig = detrend(sig)

    # Z-normalise
    std = sig.std()
    if std < 1e-6:
        return None
    sig = (sig - sig.mean()) / std

    filtered = bandpass_filter(sig, fps)

    # Peak detection with adaptive parameters
    min_dist = int(fps * 0.35)  # minimum ~170 BPM
    peaks, properties = find_peaks(
        filtered, distance=min_dist, prominence=0.2, height=0.0
    )

    if len(peaks) < 2:
        return None

    # Use inter-peak intervals for more accurate BPM
    intervals = np.diff(peaks) / fps  # seconds between peaks
    median_interval = np.median(intervals)

    if median_interval <= 0:
        return None

    bpm = 60.0 / median_interval

    # Clamp to physiological range
    bpm = max(45.0, min(170.0, bpm))
    return round(bpm, 1)


# ---------------------------------------------------------------------------
# 4. Stress estimation via HRV (RMSSD)
# ---------------------------------------------------------------------------
def calc_stress(green_buffer, fps):
    """Return (stress_label, rmssd) from the green-channel buffer."""
    min_samples = int(fps * 5)  # 5 seconds minimum
    if len(green_buffer) < min_samples:
        return ("Measuring...", 0.0)

    sig = np.array(green_buffer, dtype=np.float64)
    sig = detrend(sig)
    std = sig.std()
    if std < 1e-6:
        return ("Measuring...", 0.0)
    sig = (sig - sig.mean()) / std

    filtered = bandpass_filter(sig, fps)
    peaks, _ = find_peaks(filtered, distance=int(fps * 0.35), prominence=0.2)

    if len(peaks) < 3:
        return ("Measuring...", 0.0)

    rr = np.diff(peaks) / fps * 1000  # RR intervals in ms
    if len(rr) < 2:
        return ("Measuring...", 0.0)

    rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2)))

    if rmssd > 40:
        return ("Relaxed", rmssd)
    elif rmssd > 20:
        return ("Moderate", rmssd)
    else:
        return ("Stressed", rmssd)


# ---------------------------------------------------------------------------
# 5. Blink-rate & fatigue estimation
# ---------------------------------------------------------------------------
def calc_blink_rate(ear_history, fps):
    """Count blinks from EAR history and return (blinks/min, fatigue_label)."""
    if len(ear_history) < fps * 3:
        return (0.0, "Measuring...")

    blink_count = 0
    consec = 0
    for ear in ear_history:
        if ear < EAR_THRESHOLD:
            consec += 1
        else:
            if 2 <= consec <= 6:  # wider window for blink detection
                blink_count += 1
            consec = 0

    duration_min = len(ear_history) / fps / 60.0
    if duration_min < 0.01:
        return (0.0, "Measuring...")

    bpm = blink_count / duration_min
    bpm = max(0.0, min(40.0, bpm))

    if bpm < 8:
        label = "Drowsy"
    elif bpm <= 12:
        label = "Fatigued"
    elif bpm <= 20:
        label = "Normal"
    else:
        label = "Hyperalert"

    return (round(bpm, 1), label)


# ---------------------------------------------------------------------------
# 6. Eye Aspect Ratio (EAR)
# ---------------------------------------------------------------------------
def get_ear(landmarks, eye_indices, w, h):
    """Compute Eye Aspect Ratio from 6 landmark indices."""
    try:
        pts = np.array(
            [
                [int(landmarks[i].x * w), int(landmarks[i].y * h)]
                for i in eye_indices
            ],
            dtype=np.float64,
        )
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        h1 = np.linalg.norm(pts[0] - pts[3])
        return (v1 + v2) / (2.0 * h1 + 1e-6)
    except Exception:
        return 0.3  # default open-eye value


# ---------------------------------------------------------------------------
# 7. Cardiac Neuro Index (CNI)
# ---------------------------------------------------------------------------
def calc_cni(bpm, rmssd, blink_rate):
    """Composite wellness score (0–100). Higher = healthier state."""
    hr_score = max(0.0, 1.0 - abs(bpm - 72) / 40.0)
    hrv_score = min(1.0, rmssd / 60.0)
    blnk_score = max(0.0, 1.0 - abs(blink_rate - 17.0) / 15.0)
    cni = (0.4 * hr_score + 0.4 * hrv_score + 0.2 * blnk_score) * 100
    return round(cni, 1)
