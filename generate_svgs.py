"""Generate SVG terminal screenshots for quantbench README."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns

# ── SVG 1: Profile Report ──

console = Console(record=True, width=90)

header = Table(show_header=False, box=None, padding=(0, 2))
header.add_column(style="bold cyan", width=22)
header.add_column()
header.add_row("Model", "Meta-Llama-3-8B-Q4_K_M.gguf")
header.add_row("Format", "gguf")
header.add_row("Parameters", "8,030,261,248")
header.add_row("Size", "4.58 GB")
header.add_row("Avg bits/weight", "4.85")
header.add_row("Compression", "6.6x vs FP32")
header.add_row("Method", "ggml")

console.print(Panel(header, title="[bold]quantbench — Profile[/bold]", border_style="blue"))

dt_table = Table(title="Dtype Distribution")
dt_table.add_column("Dtype", style="bold")
dt_table.add_column("Fraction", justify="right")
dt_table.add_column("Bar", min_width=30)

data = [("q4_k_m", 0.782), ("q6_k", 0.112), ("f16", 0.086), ("q5_k_m", 0.020)]
for dtype, frac in data:
    bar_len = int(frac * 30)
    bar = "█" * bar_len + "░" * (30 - bar_len)
    console.print()  # just for spacing
    dt_table.add_row(dtype, f"{frac:.1%}", f"[green]{bar}[/green]")

console.print(dt_table)
console.print()

q_table = Table(show_header=False, box=None, padding=(0, 2))
q_table.add_column(style="bold cyan", width=22)
q_table.add_column()
q_table.add_row("Perplexity delta", "+0.0800")
q_table.add_row("Quality score", "0.8466")
q_table.add_row("Risk level", "[yellow]MEDIUM[/yellow]")
console.print(Panel(q_table, title="[bold]Quality Estimate[/bold]", border_style="yellow"))
console.print("  • Good balance of size and quality for most use cases")
console.print("  • Model: ~8.0B params, 4.6 GB at 4.9 bpw")

svg = console.export_svg(title="quantbench profile")
with open("/data/users/zacharie/repogen/quantbench/assets/profile.svg", "w") as f:
    f.write(svg)
print(f"profile.svg: {len(svg):,} bytes")

# ── SVG 2: Layerwise Analysis ──

console2 = Console(record=True, width=90)

table = Table(title="Layerwise Analysis — Sensitivity & Quantization")
table.add_column("Layer", style="bold")
table.add_column("Params", justify="right")
table.add_column("BPW", justify="right")
table.add_column("Dtype")
table.add_column("Sensitivity", justify="right")

layers = [
    ("token_embd", "262,144,000", "16.00", "f16", 0.95, "red"),
    ("blk.0", "67,174,400", "4.85", "q4_k_m", 0.78, "red"),
    ("blk.1", "67,174,400", "4.85", "q4_k_m", 0.65, "yellow"),
    ("blk.2", "67,174,400", "4.85", "q4_k_m", 0.50, "yellow"),
    ("...", "", "", "", 0, ""),
    ("blk.30", "67,174,400", "4.85", "q4_k_m", 0.50, "yellow"),
    ("blk.31", "67,174,400", "4.85", "q4_k_m", 0.60, "yellow"),
    ("output", "262,144,000", "16.00", "f16", 0.92, "red"),
]
for name, params, bpw, dtype, sens, color in layers:
    if name == "...":
        table.add_row("[dim]...[/dim]", "[dim]...[/dim]", "[dim]...[/dim]", "[dim]...[/dim]", "[dim]...[/dim]")
    else:
        table.add_row(name, params, bpw, dtype, f"[{color}]{sens:.3f}[/{color}]")

console2.print(table)
console2.print()

rec = Table(title="Mixed-Quant Recommendation (target: 4.5 bpw)")
rec.add_column("Property", style="bold")
rec.add_column("Value")
rec.add_row("High precision layers", "[cyan]6[/cyan] (embed, output, blk.0-3)")
rec.add_row("Low precision layers", "[green]28[/green] (blk.4-31)")
rec.add_row("High quant", "Q5_K_M (5.5 bpw)")
rec.add_row("Low quant", "Q4_K_M (4.85 bpw)")
rec.add_row("Estimated avg BPW", "[bold]4.48[/bold]")
console2.print(rec)

svg2 = console2.export_svg(title="quantbench layers")
with open("/data/users/zacharie/repogen/quantbench/assets/layerwise.svg", "w") as f:
    f.write(svg2)
print(f"layerwise.svg: {len(svg2):,} bytes")
