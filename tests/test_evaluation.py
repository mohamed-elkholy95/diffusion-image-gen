import pytest
import numpy as np
from src.evaluation import generate_report

class TestReport:
    def test_output(self):
        report = generate_report({"loss": [0.5, 0.3]}, n_images=10)
        assert "# Diffusion" in report
        assert "10" in report
