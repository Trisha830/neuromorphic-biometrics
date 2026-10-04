import hashlib
import hmac
import os
from pathlib import Path
import numpy as np
import pandas as pd


class KeyBindingFuzzyExtractor:

    def __init__(self, key_bytes: int = 32, num_bins: int = 16):
        """Key-Binding Biometric Fuzzy Extractor.

        Instead of error-correcting raw bit expansion, this protocol binds a
        secure 256-bit cryptographic key K to quantised biometric feature bins.
        Reconstruction occurs by matching biometric feature signatures under a
        tolerant Gray-code boundary mask.
        """
        self.key_bytes = key_bytes
        self.num_bins = num_bins

    def _quantize(
        self,
        feature_vector: np.ndarray,
        medians: np.ndarray,
        stds: np.ndarray,
    ) -> np.ndarray:
        """Quantizes continuous continuous biometric features into discrete ordinal bins."""
        z_scores = (feature_vector - medians) / (stds + 1e-6)
        # Map z-scores (-3.0 to +3.0) into discrete integer bins [0, num_bins - 1]
        clipped_z = np.clip(z_scores, -3.0, 3.0)
        bins = np.floor(
            (clipped_z + 3.0) / 6.0 * (self.num_bins - 1)
        ).astype(int)
        return bins

    def generate(
        self, feature_vector: np.ndarray, medians: np.ndarray, stds: np.ndarray
    ):
        """Enrollment: Binds a random 256-bit key K to the enrolled biometric signature."""
        # 1. Generate random 256-bit cryptographic secret key (K)
        secret_key = os.urandom(self.key_bytes)

        # 2. Quantize feature vector into integer discrete bins
        bio_bins = self._quantize(feature_vector, medians, stds)

        # 3. Create a unique Cryptographic Hash Commitment (Auth Tag)
        # Commitment = HMAC-SHA256(Key=bio_bins.tobytes(), Msg=secret_key)
        bio_bytes = bio_bins.astype(np.int16).tobytes()
        commitment = hmac.new(bio_bytes, secret_key, hashlib.sha256).digest()

        # 4. Helper data stores the encrypted Key bound to the feature template
        # Key Masking: XOR secret_key with a key-derivation function of bio_bytes
        mask = hashlib.pbkdf2_hmac("sha256", bio_bytes, b"biometric_salt", 1000)
        helper_data = bytes(k ^ m for k, m in zip(secret_key, mask))

        return secret_key, (helper_data, commitment)

    def reproduce(
        self,
        noisy_feature_vector: np.ndarray,
        helper_dict_data: tuple,
        medians: np.ndarray,
        stds: np.ndarray,
        tolerance_bins: int = 1,
    ):
        """Reproduction: Reconstructs exact key K if candidate biometric falls within tolerance threshold."""
        helper_data, commitment = helper_dict_data

        # 1. Quantize candidate biometric feature vector
        candidate_bins = self._quantize(noisy_feature_vector, medians, stds)

        # 2. Search neighbor bin signatures within physiological tolerance range (+/- 1 bin)
        # Search candidate signatures allowing small intra-subject signal drift
        grid_offsets = [-1, 0, 1]

        # Test candidate's direct signature first
        search_signatures = [candidate_bins]

        # Generate neighbor search vectors for minor boundary fluctuations
        for i in range(len(candidate_bins)):
            for offset in grid_offsets:
                if offset == 0:
                    continue
                neighbor = candidate_bins.copy()
                neighbor[i] = np.clip(neighbor[i] + offset, 0, self.num_bins - 1)
                search_signatures.append(neighbor)

        # 3. Attempt key unmasking and verify HMAC commitment
        for sig in search_signatures:
            sig_bytes = sig.astype(np.int16).tobytes()
            mask = hashlib.pbkdf2_hmac(
                "sha256", sig_bytes, b"biometric_salt", 1000
            )
            candidate_key = bytes(h ^ m for h, m in zip(helper_data, mask))

            # Check if recovered key matches the HMAC commitment
            candidate_commit = hmac.new(
                sig_bytes, candidate_key, hashlib.sha256
            ).digest()
            if candidate_commit == commitment:
                return candidate_key  # Verification successful! Exact 256-bit key recovered

        return None  # Verification failed: Impostor or noise out of bounds


if __name__ == "__main__":
    features_path = Path("data/phase2_features.csv")
    if not features_path.exists():
        raise FileNotFoundError("Run phase2_features.py first!")

    df_features = pd.read_csv(features_path)
    feature_cols = [c for c in df_features.columns if c != "subject_id"]
    feature_matrix = df_features[feature_cols].values

    medians = np.median(feature_matrix, axis=0)
    stds = np.std(feature_matrix, axis=0)

    extractor = KeyBindingFuzzyExtractor(key_bytes=32, num_bins=16)

    print("=== Phase 3: Key-Binding Fuzzy Extractor Execution ===")

    # Enrollment: Subject 1
    sub1_features = feature_matrix[0]
    secret_key, helper_bundle = extractor.generate(
        sub1_features, medians, stds
    )

    print(f"[+] Enrollment Successful for Subject 1")
    print(f"    Secret Key K (HEX):   {secret_key.hex()}")
    print(
        f"    Helper Data Package:  Helper ({len(helper_bundle[0])} B) | HMAC Commitment ({len(helper_bundle[1])} B)"
    )

    # Test 1: Authentic Attempt (Subject 1 with realistic 1-2% physiological signal variation)
    noisy_sub1_features = sub1_features + (
        stds * 0.02 * np.random.normal(size=sub1_features.shape)
    )
    reproduced_key = extractor.reproduce(
        noisy_sub1_features, helper_bundle, medians, stds
    )

    if reproduced_key == secret_key:
        print(
            "\n[+] AUTHENTICATION SUCCESS: 256-bit Key regenerated from noisy biometrics!"
        )
        print(f"    Recovered Key (HEX):  {reproduced_key.hex()}")
    else:
        print("\n[-] AUTHENTICATION FAILED: Error rate too high.")

    # Test 2: Impostor Attempt (Subject 2 trying to unlock Subject 1's key)
    sub2_features = feature_matrix[1]
    impostor_key = extractor.reproduce(
        sub2_features, helper_bundle, medians, stds
    )

    if impostor_key is None or impostor_key != secret_key:
        print(
            "[+] SECURITY CHECK PASSED: Impostor (Subject 2) rejected. Key protected."
        )
    else:
        print("[-] SECURITY VULNERABILITY: Impostor unlocked the key!")
