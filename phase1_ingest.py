from pathlib import Path
import numpy as np
import pandas as pd


def generate_synthetic_gazebase(
    num_subjects: int = 5,
    samples_per_subject: int = 10000,
    sampling_rate: int = 1000,
) -> pd.DataFrame:
    """Generates synthetic high-frequency gaze and pupillometry time-series

    mimicking GazeBase dynamics (1000 Hz) for testing Phase 1 to Phase 4.
    """
    np.random.seed(42)
    records = []

    for sub_id in range(1, num_subjects + 1):
        # Baseline eye dynamics per individual (biometric uniqueness)
        sub_saccade_gain = np.random.uniform(0.8, 1.2)
        sub_pupil_base = np.random.uniform(3.0, 5.0)

        time = np.arange(samples_per_subject) / sampling_rate

        # Simulate horizontal (x) and vertical (y) gaze coordinates with saccades
        x_pos = np.sin(2 * np.pi * 0.5 * time) * 10 * sub_saccade_gain
        y_pos = np.cos(2 * np.pi * 0.5 * time) * 5 * sub_saccade_gain

        # Add high-frequency micro-fixation jitter
        x_pos += np.random.normal(0, 0.05, samples_per_subject)
        y_pos += np.random.normal(0, 0.05, samples_per_subject)

        # Compute numerical velocity profiles (deg/s)
        x_vel = np.gradient(x_pos, time)
        y_vel = np.gradient(y_pos, time)

        # Simulate pupil diameter dynamics (mm) with autonomic noise
        pupil = (
            sub_pupil_base
            + 0.3 * np.sin(2 * np.pi * 2.0 * time)
            + np.random.normal(0, 0.02, samples_per_subject)
        )

        df_sub = pd.DataFrame(
            {
                "subject_id": sub_id,
                "time": time,
                "x_pos": x_pos,
                "y_pos": y_pos,
                "x_vel": x_vel,
                "y_vel": y_vel,
                "pupil_diameter": pupil,
                "round": 1,
                "session": 1,
                "task": "Fixation",
            }
        )
        records.append(df_sub)

    master_df = pd.concat(records, ignore_index=True)
    return master_df


if __name__ == "__main__":
    print("[*] Generating synthetic GazeBase time-series dataset...")
    df_sample = generate_synthetic_gazebase()

    output_file = Path("data/preprocessed_phase1_sample.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_sample.to_csv(output_file, index=False)

    print(f"[+] Success! Dataset exported to: {output_file.resolve()}")
    print(f"[+] Output DataFrame Shape: {df_sample.shape}")
    print("\n--- Sample Preprocessed Data ---")
    print(
        df_sample[
            ["subject_id", "time", "x_vel", "y_vel", "pupil_diameter"]
        ].head(10)
    )
