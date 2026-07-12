# Testing

## Regression Tests

Run

python Tests/run_tests.py

before modifying acquisition code.

Regression tests use synthetic data and verify software correctness.

---

## Session Validation

Run

python Tools/validate_session.py

after every experimental recording.

Validation analyzes completed recordings.

Validation never modifies recorded data.

---

## Expected Outputs

run_tests.py

- Console summary
- Pass/fail status

validate_session.py

- Console summary
- eyetracker_validation.txt