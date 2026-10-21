"""Entry point `ingest` del MLflow Project (ver `MLproject` en la raíz).

Ejecuta la ingesta y el feature engineering de una pregunta (o de las tres)
invocando los módulos ya existentes de cada pregunta, en el mismo orden que
los targets `data` / `data_p1` / `data_p3` del Makefile.

Uso:
    python scripts/ingest.py --question all   # por defecto: P1 + P2 + P3
    python scripts/ingest.py --question p3

Vía MLflow Project:
    mlflow run . -e ingest -P question=p3
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Mismos módulos que invocan los targets `data_p1`/`data`(p2)/`data_p3` del Makefile.
PIPELINES = {
    "p1": ["src.data.p1_dataset", "src.features.p1_features"],
    "p2": ["src.data.p2_dataset", "src.features.p2_features"],
    "p3": ["src.data.dataset", "src.features.features"],
}


def run_module(module: str) -> None:
    print(f"\n==> python -m {module}")
    subprocess.run([sys.executable, "-m", module], cwd=ROOT, check=True)


def run_ingest(question: str) -> None:
    preguntas = PIPELINES if question == "all" else {question: PIPELINES[question]}
    for nombre, modulos in preguntas.items():
        print(f"\n{'=' * 70}\nIngesta de {nombre.upper()}\n{'=' * 70}")
        for modulo in modulos:
            run_module(modulo)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--question",
        choices=[*PIPELINES, "all"],
        default="all",
        help="Pregunta a ingerir (p1, p2 o p3) o 'all' para las tres (default).",
    )
    args = parser.parse_args()
    run_ingest(args.question)


if __name__ == "__main__":
    main()
