"""
Punto de Entrada General de Entrenamiento para el Proyecto (Equipo 30).
=======================================================================
Permite entrenar y registrar todos los experimentos (P1, P2 y P3) en MLflow
o ejecutar selectivamente el pipeline deseado.

Ejecución desde CLI:
    python -m src.models.train
"""

import typer


app = typer.Typer(help="CLI de Entrenamiento y Registro en MLflow - MiBici Equipo 30")


@app.command()
def all():
    """Ejecuta y registra en MLflow todos los experimentos (P1, P2 y P3)."""
    from src.models.run_all_experiments import main as run_all

    run_all()


@app.command()
def p2():
    """Ejecuta y registra en MLflow el pipeline específico de Pregunta 2."""
    from src.models.p2_train import train_and_evaluate as run_p2

    run_p2()


@app.command()
def p1():
    """Entrena P1 con características preparadas y registra corridas en MLflow."""
    from src.models.p1_train import run_pipeline

    run_pipeline()


if __name__ == "__main__":
    app()
