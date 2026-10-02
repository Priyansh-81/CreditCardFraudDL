"""Direct test runner executing all 10 unit and integration tests."""

import time
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_pipeline import (
    create_dummy_transaction_df,
    test_dataset_loading_and_missing_file_error,
    test_required_columns_validation,
    test_chronological_split,
    test_no_overlap_between_partitions,
    test_scaler_fitted_only_on_training_data,
    test_correct_feature_target_separation,
    test_autoencoder_forward_pass,
    test_output_shape_equals_input_shape,
    test_reconstruction_error_calculation,
    test_threshold_selection_and_application,
)


def run_all_tests():
    print("\n" + "=" * 75)
    print("EXECUTING CREDIT CARD FRAUD DETECTION TEST SUITE (10 TESTS)")
    print("=" * 75)

    df_dummy = create_dummy_transaction_df()

    tests = [
        ("1. Dataset loading & missing file guidance", lambda: test_dataset_loading_and_missing_file_error()),
        ("2. Required columns validation", lambda: test_required_columns_validation(df_dummy)),
        ("3. Chronological split ordering", lambda: test_chronological_split(df_dummy)),
        ("4. Zero overlap / temporal leakage check", lambda: test_no_overlap_between_partitions(df_dummy)),
        ("5. Scaler fitted ONLY on training data", lambda: test_scaler_fitted_only_on_training_data(df_dummy)),
        ("6. Correct feature/target separation", lambda: test_correct_feature_target_separation(df_dummy)),
        ("7. Autoencoder forward pass", lambda: test_autoencoder_forward_pass()),
        ("8. Output shape equals input shape", lambda: test_output_shape_equals_input_shape()),
        ("9. Reconstruction error calculation", lambda: test_reconstruction_error_calculation()),
        ("10. Threshold selection & application", lambda: test_threshold_selection_and_application()),
    ]

    passed = 0
    start_total = time.time()

    for name, test_func in tests:
        t0 = time.time()
        try:
            test_func()
            elapsed = time.time() - t0
            print(f"  [PASS] {name:<48} ({elapsed:.3f}s)")
            passed += 1
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [FAIL] {name:<48} ({elapsed:.3f}s)")
            print(f"         Error: {e}")

    total_time = time.time() - start_total
    print("-" * 75)
    print(f"Result: {passed}/{len(tests)} tests passed in {total_time:.2f}s")
    print("=" * 75 + "\n")

    if passed != len(tests):
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
