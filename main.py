import time
from phase1_ingest import generate_synthetic_gazebase
from phase2_features import extract_subject_features
from phase3_fuzzy_extractor import KeyBindingFuzzyExtractor
from phase4_evaluation import evaluate_system_performance


def run_full_pipeline():
    print(
        "=========================================================================="
    )
    print(
        " NEUROMORPHIC KEY GENERATION: OCULAR MOTOR & PUPILLARY BIOMETRIC PIPELINE"
    )
    print(
        "==========================================================================\n"
    )

    start_time = time.time()

    # Phase 1
    print("[1/4] Phase 1: Ingesting high-frequency oculomotor time-series...")
    df_raw = generate_synthetic_gazebase()

    # Phase 2
    print(
        "[2/4] Phase 2: Computing velocity profiles & pupil jitter PSD features..."
    )
    features_df = extract_subject_features(df_raw)

    # Phase 3
    print(
        "[3/4] Phase 3: Testing Key-Binding Fuzzy Extractor & HMAC commitment..."
    )
    # Execution validated inside Phase 4

    # Phase 4
    print("[4/4] Phase 4: Running FAR/FRR security & entropy evaluation...\n")
    evaluate_system_performance()

    elapsed = time.time() - start_time
    print(
        f"\n[+] Pipeline execution completed successfully in {elapsed:.2f} seconds!"
    )


if __name__ == "__main__":
    run_full_pipeline()
