#!/usr/bin/env python3
"""
Report generation module
Generates training reports - but we usually just read the logs
"""
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

# Constants
REPORT_VERSION = '1.0'
DEFAULT_OUTPUT_DIR = 'outputs/reports'


def generate_report(training_log: dict, output_path: str = None) -> str:
    """Generate a text report from training log"""

    lines = [
        "=" * 60,
        "TRAINING REPORT",
        f"Generated: {datetime.now().isoformat()}",
        "=" * 60,
        "",
    ]

    if 'config' in training_log:
        lines.append("Configuration:")
        for k, v in training_log['config'].items():
            lines.append(f"  {k}: {v}")
        lines.append("")

    if 'final_accuracy' in training_log:
        lines.append(f"Final Accuracy: {training_log['final_accuracy']:.4f}")

    if 'best_accuracy' in training_log:
        lines.append(f"Best Accuracy: {training_log['best_accuracy']:.4f}")

    if 'total_epochs' in training_log:
        lines.append(f"Total Epochs: {training_log['total_epochs']}")

    lines.append("")
    lines.append("=" * 60)

    report = "\n".join(lines)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)

    return report


def generate_json_report(training_log: dict, output_path: str = None) -> dict:
    """Generate JSON report - duplicate of training log basically"""
    report = {
        'version': REPORT_VERSION,
        'generated_at': datetime.now().isoformat(),
        'summary': {},
        'details': training_log,
    }

    if 'final_accuracy' in training_log:
        report['summary']['final_accuracy'] = training_log['final_accuracy']
    if 'best_accuracy' in training_log:
        report['summary']['best_accuracy'] = training_log['best_accuracy']

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

    return report


def generate_markdown_report(training_log: dict, output_path: str = None) -> str:
    """Generate markdown report - rarely used"""
    lines = [
        "# Training Report",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Summary",
        "",
    ]

    if 'final_accuracy' in training_log:
        lines.append(f"- **Final Accuracy**: {training_log['final_accuracy']:.4f}")
    if 'best_accuracy' in training_log:
        lines.append(f"- **Best Accuracy**: {training_log['best_accuracy']:.4f}")

    lines.append("")
    lines.append("## Details")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(training_log, indent=2))
    lines.append("```")

    report = "\n".join(lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)

    return report


class ReportGenerator:
    """Report generator class - overkill for our needs"""

    def __init__(self, output_dir: str = DEFAULT_OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._templates = {}

    def generate(self, log: dict, format: str = 'text') -> str:
        """Generate report in specified format"""
        if format == 'text':
            return generate_report(log, str(self.output_dir / 'report.txt'))
        elif format == 'json':
            return generate_json_report(log, str(self.output_dir / 'report.json'))
        elif format == 'markdown':
            return generate_markdown_report(log, str(self.output_dir / 'report.md'))
        else:
            raise ValueError(f"Unknown format: {format}")

    def generate_all(self, log: dict):
        """Generate all formats - nobody uses this"""
        for fmt in ['text', 'json', 'markdown']:
            self.generate(log, fmt)


# Unused functions
def old_report_generator():
    """Old report generator - deprecated"""
    pass


def experimental_report():
    """Experimental report format - WIP"""
    pass


def _internal_helper():
    """Internal helper - unused"""
    pass
