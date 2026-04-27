# V1 Change Log

Concise notes for implementation changes that affect paper claims, results, or methodology.

## 2026-04-27 - Calibrate PC+greedy mean-shift threshold

Changed:
- Replaced the fixed PC+greedy shift threshold `0.5` with a per-variable normal-approximation threshold for the difference of observational and interventional means.
- Preserved the existing greedy behavior: significant shifts orient `target -> neighbor`; non-significant shifts default to `neighbor -> target`.

Why:
- The fixed threshold was not comparable across levels because sample size and variance change across the ladder.
- The calibrated threshold makes the active PC baseline less dependent on a magic constant while keeping the experiment delta small.

Paper impact:
- The PC+greedy method description should report the threshold formula when the paper is ready to update.
- Existing PC+greedy headline numbers are stale until the active baseline is rerun.
- The reverse-on-nonshift rule should be described as a heuristic, not as proof of the reverse direction.
- On the v0 ladder seed map, the calibrated PC+greedy audit improved overall active directed F1 from `42.7` to `48.8` and SHD from `4.792` to `4.479`.

Follow-up:
- Rerun PC+greedy on the v1 ladder before updating result tables.
- Consider an abstention-aware or likelihood-ratio orientation policy separately; do not mix that with this threshold change.
