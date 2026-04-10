"""Report formatting for quantbench — text, rich, markdown, JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from quantbench._types import ModelProfile, QualityEstimate


def report_to_dict(profile: ModelProfile, quality: QualityEstimate | None = None) -> Dict[str, Any]:
    """Convert a profile and optional quality estimate to a dict."""
    d = profile.to_dict()
    if quality:
        d["quality"] = {
            "estimated_perplexity_delta": quality.estimated_perplexity_delta,
            "quality_score": quality.quality_score,
            "risk_level": quality.risk_level,
            "sensitive_layers": quality.sensitive_layers,
            "recommendations": quality.recommendations,
        }
    return d


def save_json(profile: ModelProfile, path: str | Path, quality: QualityEstimate | None = None) -> None:
    """Save a profile report as JSON."""
    Path(path).write_text(json.dumps(report_to_dict(profile, quality), indent=2, ensure_ascii=False))


def load_json(path: str | Path) -> Dict[str, Any]:
    """Load a JSON report."""
    return json.loads(Path(path).read_text())  # type: ignore[no-any-return]


def format_report_text(profile: ModelProfile, quality: QualityEstimate | None = None) -> str:
    """Format a profile as plain text."""
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("QUANTIZATION PROFILE")
    lines.append("=" * 60)
    lines.append(f"  Model:              {profile.name}")
    lines.append(f"  Format:             {profile.format.value}")
    lines.append(f"  Parameters:         {profile.total_params:,}")
    lines.append(f"  Size:               {profile.size_gb:.2f} GB")
    lines.append(f"  Avg bits/weight:    {profile.quant.avg_bits_per_weight:.2f}")
    lines.append(f"  Compression ratio:  {profile.compression_ratio:.1f}x")
    lines.append(f"  Method:             {profile.quant.method.value}")
    lines.append(f"  Tensors:            {len(profile.tensors)}")
    lines.append(f"  Layers:             {len(profile.layers)}")
    lines.append(f"  Quantized tensors:  {profile.quant.n_quantized_layers}")
    lines.append(f"  FP tensors:         {profile.quant.n_full_precision_layers}")

    if profile.quant.dtype_distribution:
        lines.append("")
        lines.append("  Dtype Distribution:")
        for dtype, frac in sorted(profile.quant.dtype_distribution.items(), key=lambda x: -x[1]):
            lines.append(f"    {dtype:12s}  {frac:.1%}")

    if quality:
        lines.append("")
        lines.append("-" * 60)
        lines.append("QUALITY ESTIMATE")
        lines.append("-" * 60)
        lines.append(f"  Perplexity delta:   +{quality.estimated_perplexity_delta:.4f}")
        lines.append(f"  Quality score:      {quality.quality_score:.4f}")
        lines.append(f"  Risk level:         {quality.risk_level}")
        if quality.recommendations:
            lines.append("")
            lines.append("  Recommendations:")
            for rec in quality.recommendations:
                lines.append(f"    • {rec}")

    return "\n".join(lines)


def format_report_rich(profile: ModelProfile, quality: QualityEstimate | None = None) -> str:
    """Format a profile using rich for terminal display. Returns rendered string."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
    except ImportError:
        return format_report_text(profile, quality)

    console = Console(record=True, width=90)

    # Header panel
    header = Table(show_header=False, box=None, padding=(0, 2))
    header.add_column(style="bold cyan", width=22)
    header.add_column()
    header.add_row("Model", profile.name)
    header.add_row("Format", profile.format.value)
    header.add_row("Parameters", f"{profile.total_params:,}")
    header.add_row("Size", f"{profile.size_gb:.2f} GB")
    header.add_row("Avg bits/weight", f"{profile.quant.avg_bits_per_weight:.2f}")
    header.add_row("Compression", f"{profile.compression_ratio:.1f}x vs FP32")
    header.add_row("Method", profile.quant.method.value)

    console.print(Panel(header, title="[bold]quantbench — Profile[/bold]", border_style="blue"))

    # Dtype distribution
    if profile.quant.dtype_distribution:
        dt_table = Table(title="Dtype Distribution")
        dt_table.add_column("Dtype", style="bold")
        dt_table.add_column("Fraction", justify="right")
        dt_table.add_column("Bar", min_width=20)
        for dtype, frac in sorted(profile.quant.dtype_distribution.items(), key=lambda x: -x[1]):
            bar_len = int(frac * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            dt_table.add_row(dtype, f"{frac:.1%}", f"[green]{bar}[/green]")
        console.print(dt_table)

    # Quality estimate
    if quality:
        risk_color = {"low": "green", "medium": "yellow", "high": "red", "critical": "bold red"}.get(
            quality.risk_level, "white"
        )
        q_table = Table(show_header=False, box=None, padding=(0, 2))
        q_table.add_column(style="bold cyan", width=22)
        q_table.add_column()
        q_table.add_row("Perplexity delta", f"+{quality.estimated_perplexity_delta:.4f}")
        q_table.add_row("Quality score", f"{quality.quality_score:.4f}")
        q_table.add_row("Risk level", f"[{risk_color}]{quality.risk_level.upper()}[/{risk_color}]")
        console.print(Panel(q_table, title="[bold]Quality Estimate[/bold]", border_style="yellow"))

        if quality.recommendations:
            for rec in quality.recommendations:
                console.print(f"  • {rec}")

    return console.export_text()


def format_markdown(profile: ModelProfile, quality: QualityEstimate | None = None) -> str:
    """Format a profile as Markdown."""
    lines: List[str] = []
    lines.append(f"# Quantization Profile: {profile.name}")
    lines.append("")
    lines.append("| Property | Value |")
    lines.append("|---|---|")
    lines.append(f"| Format | {profile.format.value} |")
    lines.append(f"| Parameters | {profile.total_params:,} |")
    lines.append(f"| Size | {profile.size_gb:.2f} GB |")
    lines.append(f"| Avg bits/weight | {profile.quant.avg_bits_per_weight:.2f} |")
    lines.append(f"| Compression | {profile.compression_ratio:.1f}x |")
    lines.append(f"| Method | {profile.quant.method.value} |")

    if profile.quant.dtype_distribution:
        lines.append("")
        lines.append("## Dtype Distribution")
        lines.append("")
        lines.append("| Dtype | Fraction |")
        lines.append("|---|---|")
        for dtype, frac in sorted(profile.quant.dtype_distribution.items(), key=lambda x: -x[1]):
            lines.append(f"| {dtype} | {frac:.1%} |")

    if quality:
        lines.append("")
        lines.append("## Quality Estimate")
        lines.append("")
        lines.append(f"- **Perplexity delta**: +{quality.estimated_perplexity_delta:.4f}")
        lines.append(f"- **Quality score**: {quality.quality_score:.4f}")
        lines.append(f"- **Risk level**: {quality.risk_level}")
        if quality.recommendations:
            lines.append("")
            lines.append("### Recommendations")
            lines.append("")
            for rec in quality.recommendations:
                lines.append(f"- {rec}")

    return "\n".join(lines)
