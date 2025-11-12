#!/usr/bin/env python3
"""
실험 실행 진입점
"""
from src.experiment import ExperimentRunner

if __name__ == "__main__":
    runner = ExperimentRunner()
    runner.run_full_experiment()

