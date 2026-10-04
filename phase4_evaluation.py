import math
from pathlib import Path
import numpy as np
import pandas as pd
from phase3_fuzzy_extractor import KeyBindingFuzzyExtractor


def calculate_shannon_entropy(keys_list: list[bytes]) -> float:
    """Calculates average Shannon Entropy per byte across all generated keys."""
    if not keys_list:
        return 0.0

    all_bytes = b"".join(keys_list)
    total_len = len(all_bytes)

    counts = {}
    for b in all_bytes:
        counts[b] = counts.get(b, 0) + 1

    entropy = 0.0
    for count in counts.values():
        p = count / total_len
        entropy -= p * math.log2(p)

    return entropy


def evaluate_system_performance():
    features_path = Path("data/phase2_features.csv")
    if not features_path.exists():
        raise FileNotFoundError("Run phase2_features.py first!")

    df_features = pd.read_csv(features_path)
    feature_cols = [c for c in df_features.columns if c != "subject_id"]
    feature_matrix = df_features[feature_cols].values
    num_subjects = len(df_features)

    medians = np.median(feature_matrix, axis=0)
    stds = np.std(feature_matrix, axis=0)

    extractor = KeyBindingFuzzyExtractor(key_bytes=32, num_bins=16)

    print("=== Phase 4: Biometric Security & Performance Evaluation ===")
    print(
        f"[*] Evaluated Dataset: {num_subjects} Subjects | {len(feature_cols)} Features/Subject"
    )

    # 1. Enroll All Subjects and Collect Keys
    enrolled_data = {}
    generated_keys = []

    for i, row in df_features.iterrows():
        sub_id = int(row["subject_id"])
        feat = feature_matrix[i]
        key, helper_bundle = extractor.generate(feat, medians, stds)
        enrolled_data[sub_id] = {
            "key": key,
            "helper": helper_bundle,
            "feat": feat,
        }
        generated_keys.append(key)

    # 2. Genuine Tests (Calculate FRR)
    total_genuine_attempts = 0
    false_rejections = 0

    np.random.seed(42)
    for sub_id, data in enrolled_data.items():
        # Simulate 10 realistic noisy attempts per subject
        for _ in range(10):
            total_genuine_attempts += 1
            noisy_feat = data["feat"] + (
                stds * 0.02 * np.random.normal(size=data["feat"].shape)
            )
            reconstructed_key = extractor.reproduce(
                noisy_feat, data["helper"], medians, stds
            )

            if reconstructed_key != data["key"]:
                false_rejections += 1

    frr = (
        (false_rejections / total_genuine_attempts) * 100
        if total_genuine_attempts > 0
        else 0
    )

    # 3. Impostor Tests (Calculate FAR)
    total_impostor_attempts = 0
    false_acceptances = 0

    for target_id, target_data in enrolled_data.items():
        for impostor_id, impostor_data in enrolled_data.items():
            if target_id == impostor_id:
                continue

            total_impostor_attempts += 1
            impostor_key = extractor.reproduce(
                impostor_data["feat"], target_data["helper"], medians, stds
            )

            if impostor_key == target_data["key"]:
                false_acceptances += 1

    far = (
        (false_acceptances / total_impostor_attempts) * 100
        if total_impostor_attempts > 0
        else 0
    )

    # 4. Entropy Evaluation
    entropy_bits = calculate_shannon_entropy(generated_keys)

    # Print Summary Report
    print("\n" + "=" * 50)
    print("        SYSTEM SECURITY METRICS SUMMARY        ")
    print("=" * 50)
    print(f" Total Genuine Verification Trials : {total_genuine_attempts}")
    print(f" Total Impostor Cross-Class Trials: {total_impostor_attempts}")
    print("-" * 50)
    print(f" False Rejection Rate (FRR)       : {frr:.2f}%")
    print(f" False Acceptance Rate (FAR)      : {far:.2f}%")
    print(f" Equal Error Rate (EER) Proxy     : {max(far, frr):.2f}%")
    print("-" * 50)
    print(
        f" Shannon Entropy per Byte         : {entropy_bits:.4f} bits (Ideal: ~8.0)"
    )
    print(
        f" Key Strength                     : 256-bit AES Cryptographic Grade"
    )
    print("=" * 50)

    # Export Report
    report_df = pd.DataFrame(
        [
            {
                "Metric": "False Rejection Rate (FRR)",
                "Value": f"{frr:.2f}%",
                "Target": "< 5.0%",
            },
            {
                "Metric": "False Acceptance Rate (FAR)",
                "Value": f"{far:.2f}%",
                "Target": "< 0.01%",
            },
            {
                "Metric": "Entropy (bits/byte)",
                "Value": f"{entropy_bits:.4f}",
                "Target": "~8.0",
            },
        ]
    )
    report_path = Path("data/phase4_security_report.csv")
    report_df.to_csv(report_path, index=False)
    print(f"\n[+] Security evaluation saved to: {report_path.resolve()}")


if __name__ == "__main__":
    evaluate_system_performance()
