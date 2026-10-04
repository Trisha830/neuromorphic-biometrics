import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from phase3_fuzzy_extractor import KeyBindingFuzzyExtractor

app = FastAPI(
    title="Neuromorphic Biometric Key Service",
    version="1.0.0",
    description="Zero-storage 256-bit key derivation from ocular motor and pupillary feature vectors.",
)

# Global Extractor Instance
extractor = KeyBindingFuzzyExtractor(key_bytes=32, num_bins=16)

# Standard reference statistics (matching Phase 4 baseline)
MEDIANS = np.array(
    [150.0, 45.0, 30.0, 5000.0, 4.0, 0.5, 0.05]
)  # 7-feature median baseline
STDS = np.array([25.0, 8.0, 5.0, 800.0, 0.6, 0.1, 0.01])


class EnrollmentRequest(BaseModel):
    subject_id: int
    features: list[float]  # 7 features


class ReproductionRequest(BaseModel):
    features: list[float]
    helper_hex: str
    commitment_hex: str


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Neuromorphic Biometric Key Generator",
    }


@app.post("/enroll")
def enroll_biometric(data: EnrollmentRequest):
    if len(data.features) != 7:
        raise HTTPException(
            status_code=400, detail="Feature vector must contain exactly 7 values."
        )

    feat_array = np.array(data.features)
    secret_key, helper_bundle = extractor.generate(feat_array, MEDIANS, STDS)

    return {
        "subject_id": data.subject_id,
        "status": "enrolled",
        "generated_key_hex": secret_key.hex(),
        "helper_bundle": {
            "helper_hex": helper_bundle["helper"].hex(),
            "commitment_hex": helper_bundle["commitment"].hex(),
        },
    }


@app.post("/authenticate")
def authenticate_biometric(data: ReproductionRequest):
    if len(data.features) != 7:
        raise HTTPException(
            status_code=400, detail="Feature vector must contain exactly 7 values."
        )

    try:
        helper_bytes = bytes.fromhex(data.helper_hex)
        commitment_bytes = bytes.fromhex(data.commitment_hex)
        helper_bundle = {
            "helper": helper_bytes,
            "commitment": commitment_bytes,
        }
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Invalid hexadecimal string encoding."
        )

    feat_array = np.array(data.features)
    reconstructed_key = extractor.reproduce(
        feat_array, helper_bundle, MEDIANS, STDS
    )

    if reconstructed_key is None:
        return {
            "authenticated": False,
            "message": "Authentication failed. Noise or impostor key mismatch.",
        }

    return {
        "authenticated": True,
        "recovered_key_hex": reconstructed_key.hex(),
    }
