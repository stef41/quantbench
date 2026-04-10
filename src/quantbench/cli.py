"""CLI for quantbench."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def _build_cli():  # type: ignore[no-untyped-def]
    try:
        import click
    except ImportError:
        raise SystemExit("CLI dependencies required: pip install quantbench[cli]")

    @click.group()
    @click.version_option(package_name="quantbench")
    def cli() -> None:
        """quantbench — quantization quality analyzer."""

    @cli.command()
    @click.argument("model_path", type=click.Path(exists=True))
    @click.option("--json-out", "-o", type=click.Path(), default=None)
    @click.option("--markdown", "-m", is_flag=True)
    def profile(model_path: str, json_out: Optional[str], markdown: bool) -> None:
        """Profile a quantized model file (GGUF or safetensors)."""
        from quantbench.predict import estimate_quality
        from quantbench.profile import profile_gguf, profile_safetensors
        from quantbench.report import (
            format_markdown,
            format_report_rich,
            format_report_text,
            save_json,
        )

        p = Path(model_path)
        if p.suffix == ".gguf":
            prof = profile_gguf(p)
        elif p.suffix == ".safetensors":
            prof = profile_safetensors(p)
        else:
            click.echo(f"Unknown format: {p.suffix}", err=True)
            raise SystemExit(1)

        quality = estimate_quality(prof)

        if markdown:
            click.echo(format_markdown(prof, quality))
        else:
            try:
                click.echo(format_report_rich(prof, quality))
            except Exception:
                click.echo(format_report_text(prof, quality))

        if json_out:
            save_json(prof, json_out, quality)
            click.echo(f"Report saved to {json_out}", err=True)

    @cli.command()
    @click.argument("model_a", type=click.Path(exists=True))
    @click.argument("model_b", type=click.Path(exists=True))
    def compare(model_a: str, model_b: str) -> None:
        """Compare two quantized model files."""
        from quantbench.compare import compare_profiles
        from quantbench.profile import profile_gguf, profile_safetensors

        def _load(path: str):  # type: ignore[no-untyped-def]
            p = Path(path)
            if p.suffix == ".gguf":
                return profile_gguf(p)
            elif p.suffix == ".safetensors":
                return profile_safetensors(p)
            else:
                click.echo(f"Unknown format: {p.suffix}", err=True)
                raise SystemExit(1)

        prof_a = _load(model_a)
        prof_b = _load(model_b)
        result = compare_profiles(prof_a, prof_b)
        click.echo(json.dumps(result, indent=2, default=str))

    @cli.command()
    @click.argument("model_path", type=click.Path(exists=True))
    def layers(model_path: str) -> None:
        """Show layerwise analysis for a quantized model."""
        from quantbench.layerwise import analyze_layers
        from quantbench.profile import profile_gguf, profile_safetensors

        p = Path(model_path)
        if p.suffix == ".gguf":
            prof = profile_gguf(p)
        elif p.suffix == ".safetensors":
            prof = profile_safetensors(p)
        else:
            click.echo(f"Unknown format: {p.suffix}", err=True)
            raise SystemExit(1)

        analysis = analyze_layers(prof)
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(title="Layerwise Analysis")
            table.add_column("Layer", style="bold")
            table.add_column("Params", justify="right")
            table.add_column("BPW", justify="right")
            table.add_column("Dtype")
            table.add_column("Sensitivity", justify="right")

            for row in analysis:
                sens = row["sensitivity"]
                sens_style = "green" if sens < 0.5 else "yellow" if sens < 0.7 else "red"
                table.add_row(
                    row["name"],
                    f"{row['n_params']:,}",
                    f"{row['avg_bits_per_weight']:.2f}",
                    row["dominant_dtype"],
                    f"[{sens_style}]{sens:.3f}[/{sens_style}]",
                )
            console.print(table)
        except ImportError:
            for row in analysis:
                click.echo(
                    f"{row['name']:30s} {row['n_params']:>12,} "
                    f"{row['avg_bits_per_weight']:>6.2f} {row['dominant_dtype']:8s} "
                    f"{row['sensitivity']:.3f}"
                )

    @cli.command()
    @click.argument("model_path", type=click.Path(exists=True))
    @click.option("--target-bpw", type=float, default=4.5)
    def recommend(model_path: str, target_bpw: float) -> None:
        """Recommend mixed-precision quantization strategy."""
        from quantbench.layerwise import recommend_mixed_quant
        from quantbench.profile import profile_gguf, profile_safetensors

        p = Path(model_path)
        if p.suffix == ".gguf":
            prof = profile_gguf(p)
        elif p.suffix == ".safetensors":
            prof = profile_safetensors(p)
        else:
            click.echo(f"Unknown format: {p.suffix}", err=True)
            raise SystemExit(1)

        result = recommend_mixed_quant(prof, target_bpw=target_bpw)
        click.echo(json.dumps(result, indent=2, default=str))

    return cli


cli = _build_cli()
