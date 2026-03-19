def get_health_insight(bpm, stress_label, cni_score, blink_rate, api_key=""):
    if cni_score is None or cni_score == 0:
        return "Collecting data to generate your health insight..."
    if cni_score >= 70:
        return f"Your vitals look healthy — {bpm} BPM with {stress_label.lower()} stress. Stay hydrated."
    elif cni_score >= 45:
        return f"Moderate load detected — {bpm} BPM, {stress_label.lower()} stress. Try slow deep breathing."
    else:
        return f"Elevated stress — {bpm} BPM, {stress_label.lower()} indicators. Take 10 slow breaths."
