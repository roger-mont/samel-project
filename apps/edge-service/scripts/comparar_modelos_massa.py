"""Compara modelos RAW → massa local com as calibrações de célula de docs/.

Lê os mesmos CSVs e o mesmo valor de patamar que scripts/gerar_curvas_bloco.py
(cauda estável de max(0, RAW_Amostra − Baseline_Bij)) e mede o erro em gramas em
três validações, todas fora da amostra de treino:

1. LOO por célula: ajusta na subida da célula sem um patamar e prevê esse patamar
   (interpolação dentro da faixa da própria célula);
2. subida → descida: ajusta na subida e prevê os patamares da descida (histerese);
3. célula nova: ajusta com as outras células e prevê uma célula que o modelo não viu.
   É o caso real: a curva de um setor vale para as 255 células que não foram
   calibradas. CSVs idênticos (mesmo SHA-1) entram uma vez só, senão a célula
   "nova" estaria no treino.

Modelos: curvas monótonas (PCHIP e linear sobre a isotônica, a atual), formas
fechadas (potência, exponencial, polinômios, Newton) e ML (GPR, SVR, k-NN,
Random Forest, Gradient Boosting, XGBoost, MLP). Na validação 3 os modelos de ML
também recebem a posição da célula (linha, coluna), para ver se aprendem uma
variação espacial.

Não escolha o modelo só pela tabela (item 18): com 8 células e ~10 patamares por
ramo, diferenças de poucos gramas estão dentro do ruído.

Uso:
    python scripts/comparar_modelos_massa.py [--json saida.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np

EDGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EDGE_ROOT / "scripts"))
sys.path.insert(0, str(EDGE_ROOT / "src"))

import gerar_curvas_bloco as gen  # noqa: E402
from math_engine.model_comparison import ALL_FITTERS, ML_FITTERS, Fitter  # noqa: E402

warnings.filterwarnings("ignore")


MODELS = ALL_FITTERS
ML = ML_FITTERS


# ----------------------------------------------------------------------------
# Dados e validações
# ----------------------------------------------------------------------------

def branch_arrays(points: dict[float, dict]) -> tuple[np.ndarray, np.ndarray]:
    ms = sorted(m for m in points if m > 0)
    return np.array([points[m]["raw_corrected"] for m in ms]), np.array(ms)


def _safe_predict(fit: Fitter, X: np.ndarray, m: np.ndarray, Xt: np.ndarray) -> np.ndarray | None:
    try:
        return fit(X, m)(Xt)
    except Exception:
        return None


def stats(err_kg: list[float]) -> dict:
    if not err_kg:
        return {"rmse_g": None, "mae_g": None, "max_g": None, "n": 0}
    e = np.array(err_kg) * 1000.0
    return {"rmse_g": round(float(np.sqrt(np.mean(e ** 2))), 1), "mae_g": round(float(np.mean(np.abs(e))), 1),
            "max_g": round(float(np.max(np.abs(e))), 1), "n": int(len(e))}


def within_cell(cells, fit: Fitter) -> tuple[list[float], list[float]]:
    loo: list[float] = []
    updown: list[float] = []
    for c in cells:
        x, m = branch_arrays(c.up)
        for i in range(len(x)):
            keep = np.arange(len(x)) != i
            p = _safe_predict(fit, x[keep], m[keep], x[i:i + 1])
            if p is not None:
                loo.append(float(p[0] - m[i]))
        xd, md = branch_arrays(c.down)
        if len(xd):
            p = _safe_predict(fit, x, m, xd)
            if p is not None:
                updown.extend((p - md).tolist())
    return loo, updown


def new_cell(cells, fit: Fitter, with_position: bool) -> tuple[list[float], dict[str, float]]:
    """Treina com as outras células (ramo UP) e prevê a célula deixada de fora."""
    errors: list[float] = []
    per_cell: dict[str, float] = {}
    for held in cells:
        X, M = [], []
        for c in cells:
            if c is held:
                continue
            x, m = branch_arrays(c.up)
            X.append(np.column_stack([x, np.full_like(x, c.row), np.full_like(x, c.column)]) if with_position else x)
            M.append(m)
        Xtr, Mtr = np.concatenate(X), np.concatenate(M)
        xt, mt = branch_arrays(held.up)
        Xt = np.column_stack([xt, np.full_like(xt, held.row), np.full_like(xt, held.column)]) if with_position else xt
        p = _safe_predict(fit, Xtr, Mtr, Xt)
        if p is None:
            continue
        e = (p - mt).tolist()
        errors.extend(e)
        per_cell[held.name] = round(float(np.sqrt(np.mean(np.square(e)))) * 1000.0, 1)
    return errors, per_cell


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", type=Path, default=gen.DOCS)
    parser.add_argument("--fracao-cauda", type=float, default=0.5)
    parser.add_argument("--json", type=Path, default=None, help="Grava o resultado completo em JSON")
    args = parser.parse_args()

    cells = gen.discover_cells(args.docs, args.fracao_cauda)
    unique, seen = [], {}
    for c in cells:
        digest = hashlib.sha1(c.path.read_bytes()).hexdigest()
        if digest in seen:
            print(f"[!] {c.name} tem o mesmo CSV que {seen[digest]}: entra uma vez só na validação 3")
            continue
        seen[digest] = c.name
        unique.append(c)
    print(f"Células: {', '.join(c.name for c in cells)} | patamares UP: "
          f"{sum(len(branch_arrays(c.up)[0]) for c in cells)} | DOWN: {sum(len(branch_arrays(c.down)[0]) for c in cells)}")

    results: dict[str, dict] = {}
    for name, fit in MODELS.items():
        loo, updown = within_cell(cells, fit)
        nc, per_cell = new_cell(unique, fit, with_position=False)
        res = {"loo": stats(loo), "up_para_down": stats(updown), "celula_nova": stats(nc),
               "celula_nova_por_celula_rmse_g": per_cell}
        if name in ML:
            ncp, per_cell_p = new_cell(unique, fit, with_position=True)
            res["celula_nova_com_posicao"] = stats(ncp)
            res["celula_nova_com_posicao_por_celula_rmse_g"] = per_cell_p
        results[name] = res

    print(f"\nErro em gramas (RMSE / erro máximo), cargas de 18 a 275 g por célula")
    print(f"{'modelo':<26} {'1. LOO na célula':>18} {'2. subida→descida':>18} {'3. célula nova':>16} {'3b. + posição':>15}")
    for name, r in results.items():
        def cell(s):
            return f"{s['rmse_g']:>7.0f} / {s['max_g']:<6.0f}" if s["rmse_g"] is not None else f"{'—':>16}"
        extra = r.get("celula_nova_com_posicao")
        print(f"{name:<26} {cell(r['loo']):>18} {cell(r['up_para_down']):>18} {cell(r['celula_nova']):>16} "
              f"{(cell(extra) if extra else '—'):>15}")

    print("\nCélula nova, RMSE por célula (g), modelo atual:")
    print("  " + ", ".join(f"{k}: {v:.0f}" for k, v in results["pchip_isotonica"]["celula_nova_por_celula_rmse_g"].items()))

    if args.json:
        args.json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[ok] {args.json}")


if __name__ == "__main__":
    main()
