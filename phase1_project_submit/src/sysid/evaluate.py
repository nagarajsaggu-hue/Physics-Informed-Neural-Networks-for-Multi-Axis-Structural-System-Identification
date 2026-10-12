"""Comparison of the PINN result against independent references.

The PINN result is compared against:

  1. LEAST SQUARES
     A diagnostic reference based on the transient data.

  2. ANSYS MODAL
     The explicitly selected physical reference mode stored by the
     modal extraction script.

Important:
For directional reduced models, the mode with the largest local nodal
displacement is not necessarily the physically selected reference mode.
If modal_results*.json contains "selected_reference_mode", that mode is
used directly.

For legacy modal files without "selected_reference_mode", the previous
local-amplitude heuristic is retained as a fallback.
"""

import json
from pathlib import Path

from . import physics


def build_table(cfg, data, identified):
    """Assemble comparison of PINN, least squares and ANSYS modal."""

    mass = cfg["physics"]["mass"]
    rows = []

    
    # PINN
    
    K_pinn = identified["identified"]["K_N_per_m"]
    C_pinn = identified["identified"]["C_N_s_per_m"]

    rows.append({
        "method": "PINN",
        "K_N_per_m": K_pinn,
        "C_N_s_per_m": C_pinn,
        "f_n_hz": identified["derived"]["f_n_hz"],
        "zeta": identified["derived"]["zeta"],
        "note": f"{identified['run']['iterations']} iterations",
    })


    # LEAST SQUARES
    
    ls = physics.least_squares_reference(data, cfg)

    rows.append({
        "method": "Least squares",
        "K_N_per_m": ls["K"],
        "C_N_s_per_m": ls["C"],
        "f_n_hz": ls["f_n_hz"],
        "zeta": ls["zeta"],
        "note": f"R^2 = {ls['r_squared']:.6f}",
    })

    
    # ANSYS MODAL
    
    modal_row = _modal_row(
        data.modal,
        mass,
        cfg,
    )

    if modal_row:
        rows.append(modal_row)

    comparison = {
        "rows": rows,
        "reference_least_squares": ls,
        "agreement": {},
    }

    
    # PINN vs least squares
    
    comparison["agreement"]["pinn_vs_least_squares"] = {
        "K_ratio":
            K_pinn / ls["K"]
            if ls["K"] else None,

        "K_percent_diff":
            100.0 * (K_pinn - ls["K"]) / ls["K"]
            if ls["K"] else None,

        "C_ratio":
            C_pinn / ls["C"]
            if ls["C"] else None,

        "C_percent_diff":
            100.0 * (C_pinn - ls["C"]) / ls["C"]
            if ls["C"] else None,
    }

    
    # PINN vs ANSYS selected modal reference
    
    if modal_row:

        Km = modal_row["K_N_per_m"]

        comparison["agreement"]["pinn_vs_ansys_modal"] = {
            "K_ratio":
                K_pinn / Km
                if Km else None,

            "K_percent_diff":
                100.0 * (K_pinn - Km) / Km
                if Km else None,
        }

    return comparison


def _modal_row(modal, mass, cfg):
    """Return the physical ANSYS modal reference.

    Preferred behaviour
    -------------------
    Use modal["selected_reference_mode"] when available.

    This is required for the Z experiment because the physical reference
    is Mode 4, while Mode 9 has the largest local |phi_Z| at Node 3575.

    Legacy behaviour
    ----------------
    For older X/Y modal JSON files that do not contain
    "selected_reference_mode", retain the previous local-amplitude
    selection logic.
    """

    if not modal:
        return None


    # 1. PREFERRED METHOD:
    #    explicit reference selected by modal extraction
    

    selected = modal.get("selected_reference_mode")

    if selected is not None:

        required = (
            "mode",
            "frequency_hz",
            "generalized_mass_kg",
            "generalized_stiffness_N_per_m",
        )

        if all(selected.get(key) is not None for key in required):

            return {
                "method": "ANSYS modal",
                "K_N_per_m":
                    selected["generalized_stiffness_N_per_m"],

                "C_N_s_per_m":
                    None,

                "f_n_hz":
                    selected["frequency_hz"],

                "zeta":
                    None,

                "note":
                    (
                        f"mode {selected['mode']}, "
                        f"m* = "
                        f"{selected['generalized_mass_kg']:,.0f} kg "
                        f"(selected reference)"
                    ),
            }

    
    # 2. FALLBACK FOR LEGACY MODAL JSON FILES
    

    if "modes" not in modal:
        return None

    obs_col = cfg["data"].get(
        "observation_column",
        "X",
    )

    if "_Y" in obs_col:

        phi_key = "phi_y"

    elif "_Z" in obs_col:

        phi_key = "phi_z"

    else:

        phi_key = "phi_x"

    usable = [
        m
        for m in modal["modes"]
        if (
            m.get(phi_key) is not None
            and abs(m[phi_key]) > 0
            and m.get(
                "generalized_mass_kg",
                float("inf"),
            ) < 1e9
        )
    ]

    if not usable:
        return None

    threshold = (
        0.5
        *
        max(
            abs(m[phi_key])
            for m in usable
        )
    )

    candidates = [
        m
        for m in usable
        if abs(m[phi_key]) >= threshold
    ]

    mode = min(
        candidates,
        key=lambda m: m["frequency_hz"],
    )

    return {
        "method": "ANSYS modal",
        "K_N_per_m":
            mode["generalized_stiffness_N_per_m"],

        "C_N_s_per_m":
            None,

        "f_n_hz":
            mode["frequency_hz"],

        "zeta":
            None,

        "note":
            (
                f"mode {mode['mode']}, "
                f"m* = "
                f"{mode['generalized_mass_kg']:,.0f} kg "
                f"(legacy fallback)"
            ),
    }


def format_table(comparison) -> str:
    """Human-readable comparison table."""

    lines = []

    head = (
        f"{'method':<16}"
        f"{'K (N/m)':>15}"
        f"{'C (N.s/m)':>15}"
        f"{'f_n (Hz)':>11}"
        f"{'zeta (%)':>11}   "
        f"note"
    )

    lines.append(head)
    lines.append("-" * len(head))

    for r in comparison["rows"]:

        k = (
            f"{r['K_N_per_m']:.4e}"
            if r["K_N_per_m"] is not None
            else "-"
        )

        c = (
            f"{r['C_N_s_per_m']:.4e}"
            if r["C_N_s_per_m"] is not None
            else "-"
        )

        fn = (
            f"{r['f_n_hz']:.4f}"
            if r["f_n_hz"] is not None
            else "-"
        )

        z = (
            f"{r['zeta'] * 100:.4f}"
            if r["zeta"] is not None
            else "-"
        )

        lines.append(
            f"{r['method']:<16}"
            f"{k:>15}"
            f"{c:>15}"
            f"{fn:>11}"
            f"{z:>11}   "
            f"{r['note']}"
        )

    lines.append("")

    
    # Least-squares diagnostic
    

    a = comparison["agreement"].get(
        "pinn_vs_least_squares",
        {},
    )

    if a.get("K_percent_diff") is not None:

        lines.append(
            f"PINN vs least squares : "
            f"K {a['K_percent_diff']:+.2f} %, "
            f"C {a['C_percent_diff']:+.2f} %"
        )

        if abs(a["K_percent_diff"]) > 5:

            lines.append(
                "  -> least-squares and PINN are not directly "
                "comparable if the PINN uses a transformed/generalized "
                "force while least squares uses the raw transient force."
            )

    
    # ANSYS validation
    

    b = comparison["agreement"].get(
        "pinn_vs_ansys_modal",
        {},
    )

    if b.get("K_percent_diff") is not None:

        lines.append(
            f"PINN vs ANSYS modal   : "
            f"K {b['K_percent_diff']:+.4f} %"
        )

        lines.append(
            "  -> ANSYS reference uses the explicitly selected "
            "physical modal reference when available."
        )

    return "\n".join(lines)


def run(cfg, data, tag="run"):
    """Load an identified result, build comparison, save and print."""

    out_dir = Path(
        cfg["output"]["dir"]
    )

    path = (
        out_dir
        /
        f"identified_{tag}.json"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"No result at {path}. "
            f"Train with tag '{tag}' first."
        )

    with open(
        path,
        encoding="utf-8",
    ) as f:

        identified = json.load(f)

    comparison = build_table(
        cfg,
        data,
        identified,
    )

    comparison_path = (
        out_dir
        /
        f"comparison_{tag}.json"
    )

    with open(
        comparison_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            comparison,
            f,
            indent=2,
        )

    text = format_table(
        comparison
    )

    print(text)

    text_path = (
        out_dir
        /
        f"comparison_{tag}.txt"
    )

    with open(
        text_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            text + "\n"
        )

    print(
        f"\nWrote {comparison_path}"
    )

    return comparison