# Changelog

All notable changes to this project are documented in this file.

## [0.2.0] - 2026-05-16

### Changed

- Added `--beta-prime` (default `0.0025`): effective beta scales with the number of input gene trees unless `--beta` is set explicitly (default `5`, unchanged from prior releases).
- Default `--emission-initialization` is `random` (was `inverse`).

## [0.0.2]

Initial public release with the Neoaves toy example, QQS I/O, and HMM-based anomaly detection.
