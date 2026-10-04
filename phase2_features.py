from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import welch


def extract_subject_features(
    df: pd.DataFrame, sampling_rate: float = 1000.0
) -> pd.DataFrame:
    """Extracts oculomotor kinematic and pupillary biometric features for each subject."""
    features_list = []

    for subject_id, group in df.groupby("subject_id"):
        # 1. Total Velocity & Acceleration Profiles
        x_vel = group["x_vel"].values
        y_vel = group["y_vel"].values
        total_vel = np.sqrt(x_vel**2 + y_vel**2)

        # Acceleration (numerical derivative of velocity)
        accel = np.gradient(total_vel, group["time"].values)

        # Kinematic Features
        peak_velocity = np.max(total_vel)
        mean_velocity = np.mean(total_vel)
        std_velocity = np.std(total_vel)
        max_acceleration = np.max(np.abs(accel))

        # 2. Pupillary Response Dynamics
        pupil = group["pupil_diameter"].values
        mean_pupil = np.mean(pupil)
        std_pupil = np.std(pupil)

        # Power Spectral Density (PSD) for Autonomic Micro-Jitter (1.5 - 3.0 Hz band)
        freqs, psd = welch(
            pupil, fs=sampling_rate, nperseg=min(1024, len(pupil))
        )
        jitter_mask = (freqs >= 1.5) & (freqs <= 3.0)
        pupil_jitter_power = (
            np.trapezoid(psd[jitter_mask], freqs[jitter_mask])
            if np.any(jitter_mask)
            else 0.0
        )

        # Combine into subject vector
        features_list.append(
            {
                "subject_id": subject_id,
                "peak_velocity": peak_velocity,
                "mean_velocity": mean_velocity,
                "std_velocity": std_velocity,
                "max_acceleration": max_acceleration,
                "mean_pupil": mean_pupil,
                "std_pupil": std_pupil,
                "pupil_jitter_power": pupil_jitter_power,
            }
        )

    features_df = pd.DataFrame(features_list)
    return features_df


if __name__ == "__main__":
    input_path = Path("data/preprocessed_phase1_sample.csv")
    if not input_path.exists():
        raise FileNotFoundError("Run phase1_ingest.py first!")

    print("[*] Loading preprocessed Phase 1 data...")
    df_raw = pd.read_csv(input_path)

    print("[*] Extracting biometric oculomotor and pupillary feature vectors...")
    features_df = extract_subject_features(df_raw)

    output_path = Path("data/phase2_features.csv")
    features_df.to_csv(output_path, index=False)

    print(f"[+] Success! Features exported to: {output_path.resolve()}")
    print("\n--- Extracted Biometric Feature Vectors ---")
    print(features_df.to_string(index=False))
