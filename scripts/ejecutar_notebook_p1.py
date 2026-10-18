"""Ejecuta el notebook de presentación con el entorno Python activo."""
from pathlib import Path
import os
import sys
import nbformat
from ipykernel.kernelspec import install
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
for variable, directory in [('JUPYTER_RUNTIME_DIR', 'jupyter-runtime'),
                             ('IPYTHONDIR', 'ipython'), ('MPLCONFIGDIR', 'matplotlib')]:
    path = root / '.venv' / directory
    path.mkdir(parents=True, exist_ok=True)
    os.environ[variable] = str(path)
install(prefix=sys.prefix, kernel_name='p1-equipo30', display_name='Python (P1 Equipo 30)')
path = root / 'notebooks/1.3-ge-p1-balance-salidas-llegadas.ipynb'
notebook = nbformat.read(path, as_version=4)
client = NotebookClient(notebook, timeout=600, kernel_name='p1-equipo30',
                        resources={'metadata': {'path': str(root)}})
try:
    client.execute()
finally:
    nbformat.write(notebook, path)
print(f'Notebook guardado con resultados: {path}')
