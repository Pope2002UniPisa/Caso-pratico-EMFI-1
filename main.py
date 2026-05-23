"""
Caso Pratico EMFI 1 – Entry point
==================================
Esegue in sequenza gli script delle tre parti del progetto.

Uso:
    python main.py          # esegue tutto (A → B → C)
    python main.py A        # solo Parte A
    python main.py B        # solo Parte B
    python main.py C        # solo Parte C
"""

import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent

PARTS = {
    "A": ROOT / "parte_A" / "parte_A.py",
    "B": ROOT / "parte_B" / "parte_B.py",
    "C": ROOT / "parte_C" / "parte_C.py",
}

def run_part(label: str) -> None:
    script = PARTS[label]
    print(f"\n{'='*60}")
    print(f"  PARTE {label}  –  {script.name}")
    print(f"{'='*60}\n")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(script.parent),
    )
    if result.returncode != 0:
        print(f"\n[ERRORE] Parte {label} terminata con codice {result.returncode}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    requested = sys.argv[1:] if len(sys.argv) > 1 else list(PARTS.keys())
    for part in requested:
        part = part.upper()
        if part not in PARTS:
            print(f"Parte '{part}' non riconosciuta. Scegli tra: {list(PARTS.keys())}")
            sys.exit(1)
        run_part(part)
    print("\nTutte le parti completate.")
