# SightBeat

**Contactless physiological monitoring from a standard webcam.**  
No wearables. No hardware. No installation. Just a browser.

Live demo → [Deploying Soon]  
Scan to try on mobile:  
[Coming Soon]

---

## What It Does

SightBeat uses your webcam to extract real-time health metrics 
by detecting microscopic color changes in your skin caused by 
blood flow — a technique called rPPG (Remote Photoplethysmography).

Everything runs locally in your browser. No video is ever 
transmitted to any server.

---

## Metrics

| Metric | Method | Scientific Basis |
|--------|--------|-----------------|
| Heart Rate (BPM) | Green channel rPPG, peak detection | de Haan & Jeanne, 2013 |
| HRV · RMSSD | Inter-peak interval analysis | Task Force ESC/NASPE, 1996 |
| Autonomic Balance | RMSSD sympathetic/parasympathetic index | Thayer et al., 2009 |
| Blink Rate | EAR formula on 6 eye landmarks | Soukupová & Čech, 2016 |
| Fatigue Level | Blink rate + HRV baseline drift composite | Craig et al., 2012 |
| Perfusion Index | R-channel AC/DC ratio | Reisner et al., 2008 |
| Emotional State | HRV + HR trend + activity state fusion | Thayer et al., 2009 |
| CNI Score (0–100) | Weighted composite: HR efficiency + HRV + blink | Shaffer & Ginsberg, 2017 |
| Detailed Inference | Local deterministic multi-metric synthesis | — |

---

## Getting Started

### Prerequisites
- Node.js 18 or higher
- A webcam (built-in or external)
- Good lighting (natural or warm white light preferred)
- Modern browser: Chrome, Edge, or Firefox


## How It Works

### Signal Pipeline
```
Webcam frame
    ↓
MediaPipe FaceMesh (478 landmarks)
    ↓
Forehead ROI extraction → mean R, G, B per frame
    ↓
CHROM algorithm (chrominance signal separation)
    ↓
Butterworth bandpass filter (0.75–2.5 Hz)
    ↓
Kalman filter (motion artifact rejection)
    ↓
Peak detection → Heart Rate + HRV
    ↓
Derived metrics: Fatigue, Emotional State, CNI Score
```
