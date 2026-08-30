"""benchmark_suite CLI — entry point `bs`."""
from __future__ import annotations

import typer

from benchmark_suite import __version__

app = typer.Typer(add_completion=False, invoke_without_command=True, pretty_exceptions_enable=False)


@app.callback()
def main() -> None:
    """Print version and exit."""
    typer.echo(__version__)


@app.command("version")
def version_cmd() -> None:
    """Print version and exit."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
