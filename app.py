#!/usr/bin/env python3
"""
Quantum Error Correction (QEC) - Interactive Qiskit Demonstration
==================================================================
Demonstrates:
  * 3-qubit bit-flip code   (X errors)
  * 3-qubit phase-flip code (Z errors)
  * Y errors and the limits of the 3-qubit code
  * Shor 9-qubit code       (X, Z, Y / general single-qubit errors)

Requirements:
    pip install qiskit qiskit-aer matplotlib pylatexenc numpy streamlit

Run:
    streamlit run qec_app_final.py

Statistical noise is simulated inside the circuit on data-qubit identity gates.
Other operations/readout are ideal in this teaching model; counts are not altered manually.
"""

from collections import Counter

import numpy as np
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error


# ============================================================================
# Configuration
# ============================================================================
IDEAL_SHOTS = 100
STAT_SHOTS = 2000
CHECK_SHOTS = 100

# Syndrome value (s1 s0 as integer) -> index of the faulty qubit.
#   s0 = parity(q0, q1),  s1 = parity(q1, q2)
#   (s0, s1) = (1, 0) -> q0, (1, 1) -> q1, (0, 1) -> q2
SYNDROME_TO_INDEX = {1: 0, 3: 1, 2: 2}

ERROR_TITLES = {"X": "Bit-flip Error (X)", "Z": "Phase Error (Z)", "Y": "Y Error (X and Z)"}
PREP_LABELS = {"Z": "|1⟩", "X": "|−⟩"}


# ============================================================================
# Noise models (errors happen inside the circuit, on 'id' gates)
# ============================================================================
def _noise_model(error):
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(error, ["id"])
    return nm


def create_x_noise_model(p):
    return _noise_model(pauli_error([("X", p), ("I", 1 - p)]))


def create_z_noise_model(p):
    return _noise_model(pauli_error([("Z", p), ("I", 1 - p)]))


def create_y_noise_model(p):
    return _noise_model(pauli_error([("Y", p), ("I", 1 - p)]))


def create_general_noise_model(p):
    return _noise_model(pauli_error([("X", p / 3), ("Y", p / 3), ("Z", p / 3), ("I", 1 - p)]))


NOISE_FACTORIES = {
    "X": create_x_noise_model,
    "Z": create_z_noise_model,
    "Y": create_y_noise_model,
    "G": create_general_noise_model,
}


# ============================================================================
# Running circuits
# ============================================================================
def run_circuit(qc, shots, noise_model=None):
    """Run on Aer. Prefer the stabilizer method when possible (all gates are
    Clifford + Pauli noise) so even the 17-qubit Shor circuit is fast."""
    last = None
    for method in ("stabilizer", "automatic"):
        try:
            sim = AerSimulator(method=method, noise_model=noise_model)
            tqc = transpile(qc, sim, optimization_level=0)
            return sim.run(tqc, shots=shots).result().get_counts()
        except Exception as exc:
            last = exc
    raise last


def parse_key(qc, key):
    """Split a counts key into {register_name: bitstring}."""
    names = [cr.name for cr in qc.cregs]
    return dict(zip(reversed(names), key.split()))


def extract_syndrome(qc, shots=CHECK_SHOTS):
    """Run a deterministic, no-noise walkthrough and return {register: integer}.

    Because the walkthrough inserts one explicit Pauli error and otherwise uses
    ideal operations, the syndrome should be identical on every shot. Refuse to
    silently pick a majority syndrome if that invariant is violated.
    """
    counts = run_circuit(qc, shots)
    if len(counts) != 1:
        raise RuntimeError(
            "Expected a deterministic syndrome in the walkthrough, but simulator "
            f"returned multiple outcomes: {counts}"
        )
    key = next(iter(counts))
    regs = parse_key(qc, key)
    return {n: int(v, 2) for n, v in regs.items() if n != "out"}


def out_strings(qc, counts):
    """Counter of final 'out' bitstrings (q_n ... q0)."""
    c = Counter()
    for key, n in counts.items():
        c[parse_key(qc, key)["out"]] += n
    return c


# ============================================================================
# Textbook-style circuit diagram (dashed ovals, |psi_k> markers,
# classically controlled corrections). Generated from the SAME Qiskit circuit
# that is simulated, so the diagram always matches what runs.
# ============================================================================
_GROUP_STYLE = {
    "ENCODE": ("Encode", "#2E8B57"),
    "DECODE": ("Decode", "#2E8B57"),
    "ERROR": (None, "#D62728"),
    "SYNDROME": ("Syndrome Measurement", "#7B4FA3"),
    "CORRECT": ("Correct Error", "#1F5FBF"),
    "MEASURE": ("Final Measurement", "#555555"),
}
_GROUP_OF = {
    # Extracting a syndrome is not itself applying a correction.
    "SYNDROME": "SYNDROME", "X-SYNDROME": "SYNDROME", "Z-SYNDROME": "SYNDROME",
    "CORRECT": "CORRECT", "ENCODE": "ENCODE", "DECODE": "DECODE",
    "ERROR": "ERROR", "MEASURE": "MEASURE",
}
_GAP = 0.9


def _layout_circuit(qc):
    """Assign every instruction to a column; labelled barriers start new sections."""
    n = qc.num_qubits
    free = [0] * n
    ops, meters = [], {}
    sections = [{"name": "PREPARE", "ops": []}]
    for inst in qc.data:
        op, name = inst.operation, inst.operation.name
        wires = [qc.find_bit(q).index for q in inst.qubits]
        if name == "barrier":
            if op.label:
                col = max(free)
                free = [col] * n
                sections.append({"name": op.label, "ops": []})
            continue
        if name == "cx":
            lo, hi = min(wires), max(wires)
            col = max(free[lo:hi + 1])
            for w in range(lo, hi + 1):
                free[w] = col + 1
        elif name == "if_else":
            col = max(free)
            free = [col + 1] * n
        elif name in ("h", "x", "y", "z", "measure") or name.endswith(("error", "fix")):
            col = free[wires[0]]
            free[wires[0]] = col + 1
        else:
            raise NotImplementedError(name)
        d = {"name": name, "wires": wires, "col": col, "sec": len(sections) - 1}
        if name == "measure":
            meters[qc.find_bit(inst.clbits[0]).index] = d
        if name == "if_else":
            reg, value = op.condition
            d["condition_register"] = getattr(reg, "name", None)
            body = op.blocks[0]
            b = body.data[0]
            d["gate"] = b.operation.name
            d["target"] = qc.find_bit(inst.qubits[body.find_bit(b.qubits[0]).index]).index
            d["value"] = value
            d["cond_bits"] = [(qc.find_bit(cb).index, (value >> i) & 1) for i, cb in enumerate(reg)]
        ops.append(d)
        sections[-1]["ops"].append(d)
    return {"ops": ops, "meters": meters, "sections": sections}


def draw_textbook_circuit(qc, title, ghost=None, error_title="Error", highlight=None,
                          footnote=None, prep_label=None):
    """
    ghost     : (pauli, [wires]) -> dashed boxes for 'possible error' locations
    highlight : syndrome value, or {classical_register_name: value} for multi-register circuits,
                whose controlled correction branches are highlighted as firing
    """
    from matplotlib.patches import Rectangle, Ellipse, Circle, Arc, FancyBboxPatch

    L = _layout_circuit(qc)
    ops, meters, sections = L["ops"], L["meters"], L["sections"]
    n = qc.num_qubits
    X = lambda d: d["col"] + _GAP * d["sec"]
    xs = [X(d) for d in ops]
    xmin, xmax = min(xs), max(xs)

    fig_w = max(11.0, (xmax - xmin + 6) * 0.62)
    fig_h = max(5.0, n * 0.62 + 3.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_aspect("equal")
    ax.axis("off")

    def box(x, y, txt, fc="white", ec="black", ls="-", tc="black", w=0.54):
        ax.add_patch(Rectangle((x - w / 2, y - w / 2), w, w, fc=fc, ec=ec, ls=ls, lw=1.7, zorder=3))
        ax.text(x, y, txt, ha="center", va="center", fontsize=13, color=tc, zorder=4)

    # Wires and initial-state labels. Only show |psi>, |0>, ... at the right
    # when this circuit actually contains the decode/final-output stage. A
    # syndrome-only circuit is still encoded and must not be labelled as decoded.
    has_decoded_output = any(cr.name == "out" for cr in qc.cregs) and any(
        section["name"] == "DECODE" and section["ops"] for section in sections
    )
    for w in range(n):
        ax.plot([xmin - 0.9, xmax + 0.9], [-w, -w], color="black", lw=1.2, zorder=1)
        bit_locations = qc.find_bit(qc.qubits[w]).registers
        reg, idx = bit_locations[0]
        ax.text(xmin - 1.05, -w, f"{reg.name}{idx}  |0⟩", ha="right", va="center", fontsize=11)
        if has_decoded_output and reg.name == "q":
            # Use the concrete test state when known; reserve |psi> for an explicitly
            # generic input (not what this app currently prepares by default).
            output_state = prep_label if prep_label else "|ψ⟩"
            ax.text(xmax + 1.05, -w, output_state if idx == 0 else "|0⟩",
                    ha="left", va="center", fontsize=11)

    # gates
    for d in ops:
        x, nm, wr = X(d), d["name"], d["wires"]
        if nm == "cx":
            c, t = wr
            ax.plot([x, x], [-c, -t], color="black", lw=1.5, zorder=2)
            ax.add_patch(Circle((x, -c), 0.09, color="black", zorder=4))
            ax.add_patch(Circle((x, -t), 0.19, fc="white", ec="black", lw=1.5, zorder=3))
            ax.plot([x - 0.19, x + 0.19], [-t, -t], color="black", lw=1.5, zorder=4)
            ax.plot([x, x], [-t - 0.19, -t + 0.19], color="black", lw=1.5, zorder=4)
        elif nm == "measure":
            y = -wr[0]
            ax.add_patch(Rectangle((x - 0.29, y - 0.27), 0.58, 0.54, fc="white", ec="black", lw=1.5, zorder=3))
            ax.add_patch(Arc((x, y - 0.1), 0.40, 0.40, theta1=0, theta2=180, lw=1.4, zorder=4))
            ax.plot([x, x + 0.12], [y - 0.1, y + 0.12], color="black", lw=1.4, zorder=4)
        elif nm == "h":
            box(x, -wr[0], "H")
        elif nm.endswith("error"):
            box(x, -wr[0], nm[0].upper(), fc="#D62728", ec="#8B0000", tc="white")
        elif nm.endswith("fix"):
            box(x, -wr[0], nm[0].upper(), fc="#2CA02C", ec="#146314", tc="white")
        elif nm in ("x", "y", "z"):
            box(x, -wr[0], nm.upper())
        elif nm == "if_else":
            y = -d["target"]
            if isinstance(highlight, dict):
                sel = highlight.get(d.get("condition_register")) == d["value"]
            else:
                sel = highlight is not None and d["value"] == highlight
            box(x, y, d["gate"].upper(), fc="#2CA02C" if sel else "white",
                ec="#146314" if sel else "black", ls="-" if sel else "--",
                tc="white" if sel else "black")
            for k, (cb, bit) in enumerate(d["cond_bits"]):
                m = meters[cb]
                mx, my = X(m), -m["wires"][0]
                off = (k - (len(d["cond_bits"]) - 1) / 2) * 0.24
                ycl = my - 0.42
                ax.plot([mx, mx, x + off, x + off], [my - 0.27, ycl, ycl, y - 0.28],
                        color="black", lw=1.5, zorder=2, solid_joinstyle="round")
                ax.text(x + off + (-0.06 if k == 0 else 0.06), y - 0.45, str(bit), fontsize=9,
                        ha="right" if k == 0 else "left", va="center",
                        color="#146314" if sel else "black")

    # dashed 'possible error' boxes on the other qubits
    if ghost:
        pauli, gw = ghost
        for d in ops:
            if d["name"].endswith("error"):
                for w in gw:
                    if w not in d["wires"]:
                        box(X(d), -w, pauli, fc="#F2F2F2", ec="#888888", ls="--", tc="#888888")

    # ovals around sections, merged by group
    groups = []
    for si, sec in enumerate(sections):
        key = _GROUP_OF.get(sec["name"])
        if not sec["ops"] or key is None:
            continue
        wires = {w for d in sec["ops"] for w in d["wires"]}
        if key == "ERROR" and ghost:
            wires |= set(ghost[1])
        if groups and groups[-1]["key"] == key and groups[-1]["last"] == si - 1:
            g = groups[-1]
            g["ops"] += sec["ops"]; g["wires"] |= wires; g["last"] = si
        else:
            groups.append({"key": key, "ops": list(sec["ops"]), "wires": wires, "last": si})
    top, bottom, left, right = 1.0, -(n - 1) - 1.0, xmin - 3.0, xmax + 2.2
    for g in groups:
        label, color = _GROUP_STYLE[g["key"]]
        if g["key"] == "ERROR":
            label = error_title
        gx = [X(d) for d in g["ops"]]
        wmin, wmax = min(g["wires"]), max(g["wires"])
        cx, cy = (min(gx) + max(gx)) / 2, -(wmin + wmax) / 2
        if wmax - wmin > 6:
            a = (max(gx) - min(gx)) / 2 + 0.6
            b = (wmax - wmin) / 2 + 0.6
            ax.add_patch(FancyBboxPatch((cx - a, cy - b), 2 * a, 2 * b,
                                        boxstyle="round,pad=0,rounding_size=0.6",
                                        fill=False, ls="--", ec=color, lw=1.4, zorder=0))
        else:
            a = ((max(gx) - min(gx)) / 2 + 0.55) * 1.32
            b = ((wmax - wmin) / 2 + 0.55) * 1.32
            ax.add_patch(Ellipse((cx, cy), 2 * a, 2 * b, fill=False, ls="--", ec=color, lw=1.4, zorder=0))
        ly = max(cy + b + 0.12, 0.85)
        ax.text(cx, ly, label, color=color, ha="center", va="bottom", fontsize=11, fontweight="bold")
        top = max(top, ly + 0.7)
        bottom = min(bottom, cy - b - 0.2)

    # section boundaries with |psi_k> markers
    k, seen_encode = 0, False
    live = [(i, s) for i, s in enumerate(sections) if s["ops"]]
    for (i1, s1), (i2, s2) in zip(live, live[1:]):
        if s1["name"] == "ENCODE":
            seen_encode = True
        if not seen_encode:
            continue
        k += 1
        bx = (max(X(d) for d in s1["ops"]) + min(X(d) for d in s2["ops"])) / 2
        ax.plot([bx, bx], [0.75, -(n - 1) - 0.75], color="#555555", ls="--", lw=1.0, zorder=0)
        ax.text(bx, -(n - 1) - 1.0, rf"$|\psi_{{{k}}}\rangle$", ha="center", va="top", fontsize=11)
        bottom = min(bottom, -(n - 1) - 1.9)

    if prep_label and sections[0]["ops"]:
        px = sum(X(d) for d in sections[0]["ops"]) / len(sections[0]["ops"])
        ax.text(px, 0.62, f"Prepare {prep_label}", ha="center", va="bottom", fontsize=10, color="#555555")

    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top + 0.3)
    unit = 0.6
    w_in = max(9.0, (right - left) * unit)
    h_in = (top + 0.3 - bottom) * unit + 1.5
    fig.set_size_inches(w_in, h_in)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.7 / h_in, top=1 - 0.8 / h_in)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1 - 0.12 / h_in)
    if footnote:
        fig.text(0.5, 0.12 / h_in, footnote, ha="center", va="bottom",
                 fontsize=10, style="italic", color="#333333")
    return fig


def _footnote(code, pauli=None):
    if pauli == "Y":
        return ("Y = iXZ: the bit-flip code corrects only the X part. The Z part is NOT detected, "
                "so a phase error remains after this correction.")
    if code == "phaseflip":
        return ("Each ancilla (H, controlled-X, H) measures a neighbouring X_i X_j parity; "
                "a Z error flips the result. The syndrome (s0 s1) selects the classically "
                "controlled Z correction.")
    return ("Dashed boxes mark possible error locations. The syndrome (s0 s1) selects the "
            "classically controlled X correction.")


# ============================================================================
# Building blocks: logical state, error injection
# ============================================================================
def prepare_logical(qc, qubit, basis):
    """basis 'Z': logical |1>.  basis 'X': logical |->.  (Both read out as 1.)"""
    qc.x(qubit)
    if basis == "X":
        qc.h(qubit)


def measure_logical(qc, qubit, clbit, basis):
    if basis == "X":
        qc.h(qubit)
    qc.measure(qubit, clbit)


def _marker_gate(pauli, kind):
    """A Pauli gate with its own name so it can be coloured in the diagram."""
    sub = QuantumCircuit(1, name=f"{pauli.lower()}{kind.lower()}")
    getattr(sub, pauli.lower())(0)
    return sub.to_gate(label=f"{pauli} {kind}")


def apply_fix(qc, gate, qubit):
    qc.append(_marker_gate(gate.upper(), "FIX"), [qubit])


def insert_error(qc, data, inject=None, noisy=False):
    """`inject` = {qubit_index: 'X'|'Y'|'Z'} places a visible error gate;
    `noisy` places 'id' gates that the NoiseModel turns into random errors."""
    qc.barrier(label="ERROR")
    for idx, pauli in (inject or {}).items():
        qc.append(_marker_gate(pauli, "ERROR"), [data[idx]])
    if noisy:
        for qb in data:
            qc.id(qb)


# ============================================================================
# 3-qubit codes
# ============================================================================
def encode_three_qubit_bit_flip(qc, q):
    qc.cx(q[0], q[1])
    qc.cx(q[0], q[2])


def encode_three_qubit_phase_flip(qc, q):
    encode_three_qubit_bit_flip(qc, q)
    qc.h(q)


def measure_x_syndrome(qc, q, anc, syn):
    qc.cx(q[0], anc[0]); qc.cx(q[1], anc[0])
    qc.cx(q[1], anc[1]); qc.cx(q[2], anc[1])
    qc.measure(anc[0], syn[0]); qc.measure(anc[1], syn[1])


def measure_z_syndrome(qc, q, anc, syn):
    for a, (i, j) in zip(anc, [(0, 1), (1, 2)]):
        qc.h(a)
        qc.cx(a, q[i]); qc.cx(a, q[j])
        qc.h(a)
    qc.measure(anc[0], syn[0]); qc.measure(anc[1], syn[1])


def _correct_repetition(qc, q, syn, gate, detected):
    if detected is None:
        for value, idx in SYNDROME_TO_INDEX.items():
            with qc.if_test((syn, value)):
                getattr(qc, gate)(q[idx])
    else:
        apply_fix(qc, gate, q[detected])


def correct_x_error(qc, q, syn, detected=None):
    _correct_repetition(qc, q, syn, "x", detected)


def correct_z_error(qc, q, syn, detected=None):
    _correct_repetition(qc, q, syn, "z", detected)


def decode_three_qubit(qc, q, code):
    if code == "phaseflip":
        qc.h(q)
    qc.cx(q[0], q[2])
    qc.cx(q[0], q[1])


def build_three_qubit_circuit(code, basis, inject=None, noisy=False,
                              stage="full", correction="dynamic", fixed=None):
    q = QuantumRegister(3, "q")
    anc = QuantumRegister(2, "anc")
    syn = ClassicalRegister(2, "syn")
    out = ClassicalRegister(3, "out")
    qc = QuantumCircuit(q, anc, syn) if stage == "syndrome" else QuantumCircuit(q, anc, syn, out)

    qc.barrier(label="PREPARE")
    prepare_logical(qc, q[0], basis)
    qc.barrier(label="ENCODE")
    (encode_three_qubit_bit_flip if code == "bitflip" else encode_three_qubit_phase_flip)(qc, q)
    insert_error(qc, q, inject, noisy)
    qc.barrier(label="SYNDROME")
    (measure_x_syndrome if code == "bitflip" else measure_z_syndrome)(qc, q, anc, syn)
    if stage == "syndrome":
        return qc

    if correction != "none":
        qc.barrier(label="CORRECT")
        fix = correct_x_error if code == "bitflip" else correct_z_error
        fix(qc, q, syn, None if correction == "dynamic" else fixed)
    qc.barrier(label="DECODE")
    decode_three_qubit(qc, q, code)
    qc.barrier(label="MEASURE")
    measure_logical(qc, q[0], out[0], basis)
    qc.measure(q[1], out[1])
    qc.measure(q[2], out[2])
    return qc


# ============================================================================
# Shor 9-qubit code
# ============================================================================
def encode_shor(qc, q):
    qc.cx(q[0], q[3]); qc.cx(q[0], q[6])
    qc.h([q[0], q[3], q[6]])
    for b in (0, 3, 6):
        qc.cx(q[b], q[b + 1]); qc.cx(q[b], q[b + 2])


def shor_measure_x_syndrome(qc, q, anc, sx):
    for k in range(3):
        b = 3 * k
        qc.cx(q[b], anc[2 * k]);         qc.cx(q[b + 1], anc[2 * k])
        qc.cx(q[b + 1], anc[2 * k + 1]); qc.cx(q[b + 2], anc[2 * k + 1])
        qc.measure(anc[2 * k], sx[k][0]); qc.measure(anc[2 * k + 1], sx[k][1])


def shor_measure_z_syndrome(qc, q, anc, sz):
    for i, (b1, b2) in enumerate([(0, 3), (3, 6)]):
        qc.h(anc[i])
        for j in list(range(b1, b1 + 3)) + list(range(b2, b2 + 3)):
            qc.cx(anc[i], q[j])
        qc.h(anc[i])
        qc.measure(anc[i], sz[i])


def decode_shor(qc, q):
    for b in (6, 3, 0):
        qc.cx(q[b], q[b + 2]); qc.cx(q[b], q[b + 1])
    qc.h([q[0], q[3], q[6]])
    qc.cx(q[0], q[6]); qc.cx(q[0], q[3])


def identify_shor_errors(syndromes):
    fixes = []
    for k in range(3):
        v = syndromes.get(f"sx{k}", 0)
        if v:
            fixes.append(("x", 3 * k + SYNDROME_TO_INDEX[v]))
    v = syndromes.get("sz", 0)
    if v:
        # The phase syndrome identifies a 3-qubit Shor block, not a specific
        # qubit index. Apply Z to the representative at the start of that block.
        fixes.append(("z", 3 * SYNDROME_TO_INDEX[v]))
    return fixes


def build_shor_circuit(basis, inject=None, noisy=False, stage="full",
                       correction="dynamic", fixed=None):
    q = QuantumRegister(9, "q")
    ax = QuantumRegister(6, "ax")
    az = QuantumRegister(2, "az")
    sx = [ClassicalRegister(2, f"sx{k}") for k in range(3)]
    sz = ClassicalRegister(2, "sz")
    out = ClassicalRegister(9, "out")
    qc = (QuantumCircuit(q, ax, az, *sx, sz)
          if stage == "syndrome"
          else QuantumCircuit(q, ax, az, *sx, sz, out))

    qc.barrier(label="PREPARE")
    prepare_logical(qc, q[0], basis)
    qc.barrier(label="ENCODE")
    encode_shor(qc, q)
    insert_error(qc, q, inject, noisy)
    qc.barrier(label="X-SYNDROME")
    shor_measure_x_syndrome(qc, q, ax, sx)
    qc.barrier(label="Z-SYNDROME")
    shor_measure_z_syndrome(qc, q, az, sz)
    if stage == "syndrome":
        return qc

    if correction != "none":
        qc.barrier(label="CORRECT")
        if correction == "dynamic":
            for k in range(3):
                for value, idx in SYNDROME_TO_INDEX.items():
                    with qc.if_test((sx[k], value)):
                        qc.x(q[3 * k + idx])
            for value, idx in SYNDROME_TO_INDEX.items():
                with qc.if_test((sz, value)):
                    qc.z(q[3 * idx])
        else:
            for gate, qubit in (fixed or []):
                apply_fix(qc, gate, q[qubit])
    qc.barrier(label="DECODE")
    decode_shor(qc, q)
    qc.barrier(label="MEASURE")
    measure_logical(qc, q[0], out[0], basis)
    for i in range(1, 9):
        qc.measure(q[i], out[i])
    return qc


# ============================================================================
# Unprotected reference qubit
# ============================================================================
def build_unprotected(basis, noisy=True):
    q = QuantumRegister(1, "q")
    out = ClassicalRegister(1, "out")
    qc = QuantumCircuit(q, out)
    prepare_logical(qc, q[0], basis)
    if noisy:
        insert_error(qc, q, None, noisy)
    qc.barrier(label="MEASURE")
    measure_logical(qc, q[0], out[0], basis)
    return qc


# ============================================================================
# Statistics
# ============================================================================
def run_statistics(code, error_type, p, tests, shots=STAT_SHOTS):
    """
    code       : 'bitflip' | 'phaseflip' | 'shor'
    error_type : 'X' | 'Z' | 'Y' | 'G'
    tests      : list of (label, protected_basis, raw_basis)
    """
    nm = NOISE_FACTORIES[error_type](p)
    results = []
    for label, pbasis, rbasis in tests:
        raw_qc = build_unprotected(rbasis)
        raw_counts = run_circuit(raw_qc, shots, nm)
        raw_errors = sum(n for k, n in raw_counts.items() if k != "1")

        if code == "shor":
            qc = build_shor_circuit(pbasis, noisy=True)
        else:
            qc = build_three_qubit_circuit(code, pbasis, noisy=True)
        counts = run_circuit(qc, shots, nm)
        detected = corrected = remaining = 0
        for key, n in counts.items():
            regs = parse_key(qc, key)
            logical_ok = regs["out"][-1] == "1"
            flagged = any(int(v, 2) != 0 for name, v in regs.items() if name != "out")
            if flagged:
                detected += n
                if logical_ok:
                    corrected += n
            if not logical_ok:
                remaining += n
        results.append(dict(label=label, raw_errors=raw_errors, detected=detected,
                            corrected=corrected, remaining=remaining))
    return dict(p=p, shots=shots, tests=results)


# ============================================================================
# Streamlit application
# ============================================================================
st.set_page_config(
    page_title="Quantum Error Correction Lab",
    page_icon="⚛️",
    layout="wide",
)

st.title("⚛️ Quantum Error Correction Lab")
st.caption(
    "Interactive Qiskit/Aer simulation of three-qubit repetition codes, the Y-error "
    "limitation, and the Shor 9-qubit code. Statistical runs model independent Pauli "
    "errors on data-qubit identity gates only; other operations and readout are ideal. "
    "Measurement outcomes are never edited manually."
)


# --- Sidebar controls -------------------------------------------------------
with st.sidebar:
    st.header("Simulation settings")

    with st.form("simulation_settings"):
        demo = st.selectbox(
            "Demonstration",
            [
                "Ideal transmission",
                "3-qubit bit-flip code (X)",
                "3-qubit phase-flip code (Z)",
                "Y error — 3-qubit limitation",
                "Shor 9-qubit code",
            ],
        )

        error_pct = st.slider(
            "Physical error probability (%)",
            min_value=1, max_value=50, value=5,
            help="Used only for the statistical noisy simulation. The walkthrough error is deterministic.",
        )

        shots = st.select_slider(
            "Shots per statistical test",
            options=[100, 250, 500, 1000, 2000, 5000],
            value=2000,
        )

        max_qubit = 8 if demo == "Shor 9-qubit code" else 2
        error_qubit = st.number_input(
            "Qubit receiving the walkthrough error",
            min_value=0, max_value=max_qubit, value=0, step=1,
            help=f"Choose q0–q{max_qubit} for the deterministic single-error walkthrough.",
        )

        shor_noise = "Y"
        if demo == "Shor 9-qubit code":
            shor_noise = st.selectbox(
                "Shor-code noise channel",
                ["X", "Z", "Y", "G"], index=2,
                format_func=lambda x: {
                    "X": "X (bit flip)",
                    "Z": "Z (phase flip)",
                    "Y": "Y (combined X + Z)",
                    "G": "General random X/Y/Z",
                }[x],
            )

        run_button = st.form_submit_button("Run simulation", type="primary", use_container_width=True)

    # If a previously selected Shor q8 is retained while switching to a 3-qubit
    # demo, clamp it to a valid data-qubit index before building the circuit.
    error_qubit = max(0, min(int(error_qubit), max_qubit))

    st.divider()
    if st.button("Back to overview", use_container_width=True):
        st.session_state["show_results"] = False
        st.rerun()

# The sidebar form submits settings only when Run simulation is clicked.
if "show_results" not in st.session_state:
    st.session_state.show_results = False
if run_button:
    st.session_state.show_results = True

p = error_pct / 100.0


# --- Helpers ----------------------------------------------------------------
def show_circuit(qc, title, **kwargs):
    st.subheader(title)
    try:
        fig = draw_textbook_circuit(qc, title, **kwargs)
        st.pyplot(fig)
        plt.close(fig)
    except Exception as exc:
        st.warning(
            f"Textbook diagram unavailable ({type(exc).__name__}: {exc}). "
            "Showing the Qiskit text circuit instead."
        )
        st.code(qc.draw(output="text", fold=120), language="text")


def show_statistics(res, title):
    st.subheader(title)
    rows = []
    for t in res["tests"]:
        raw_rate = 100 * t["raw_errors"] / res["shots"]
        post_rate = 100 * t["remaining"] / res["shots"]
        success = (100 * t["corrected"] / t["detected"]) if t["detected"] else None
        rows.append({
            "Test": t["label"],
            "Unprotected logical readout errors": t["raw_errors"],
            "Unprotected readout error rate (%)": round(raw_rate, 2),
            "Syndrome-flagged shots": t["detected"],
            "Flagged shots with correct logical readout": t["corrected"],
            "Logical readout errors after QEC": t["remaining"],
            "Post-QEC logical readout error rate (%)": round(post_rate, 2),
            "Correct readout among flagged shots (%)": round(success, 2) if success is not None else "N/A",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    for row in rows:
        raw = row["Unprotected readout error rate (%)"]
        post = row["Post-QEC logical readout error rate (%)"]
        if post < raw:
            st.success(
                f'{row["Test"]}: the measured logical-readout error rate fell from '
                f"{raw:.2f}% to {post:.2f}%."
            )
        elif post > raw:
            st.warning(
                f'{row["Test"]}: the measured logical-readout error rate rose from '
                f"{raw:.2f}% to {post:.2f}%. At this physical error rate, multiple "
                "errors or the simplified noise model may outweigh the protection."
            )
        else:
            st.info(f'{row["Test"]}: no measured change in logical-readout error rate.')


def walkthrough_comparison_table(code, inject, found, bases, n_bits):
    """Deterministic comparison: same injected error, with and without correction."""
    expected = "0" * (n_bits - 1) + "1"
    rows = []
    for basis in bases:
        for label, corr in [("Without correction", "none"), ("With syndrome-controlled correction", "dynamic")]:
            if code == "shor":
                qc = build_shor_circuit(basis, inject, correction=corr, fixed=found)
            else:
                qc = build_three_qubit_circuit(code, basis, inject, correction=corr, fixed=found)
            counts = out_strings(qc, run_circuit(qc, CHECK_SHOTS))
            bits, n = counts.most_common(1)[0]
            if bits == expected:
                verdict = "Decoded output matches expected bitstring"
            elif bits[-1] == "1":
                verdict = "Logical readout correct; other qubits differ"
            else:
                verdict = "Logical readout differs from expected value"
            rows.append({
                "Basis": "0/1" if basis == "Z" else "+/-",
                "Mode": label,
                "Output": bits,
                "Verdict": verdict,
            })
    st.subheader("Deterministic single-error comparison")
    st.dataframe(rows, use_container_width=True, hide_index=True)


# --- Main body --------------------------------------------------------------
if not st.session_state.show_results:
    st.markdown(
        """
        ### About this lab

        This interface follows the standard QEC workflow:

        **prepare → encode → inject/simulate noise → measure syndrome → correct → decode → measure**

        - **Ideal transmission:** no-noise baseline; reference for all later runs.
        - **3-qubit bit-flip code:** protects against a single X error.
        - **3-qubit phase-flip code:** protects against a single Z error in the Hadamard basis.
        - **Y-error demonstration:** shows the limitation of the bit-flip repetition code
          (it corrects the X part of Y but not the Z part).
        - **Shor 9-qubit code:** combines bit-flip and phase-flip protection to correct
          arbitrary single-qubit Pauli errors under the idealized circuit model.

        **Important scope note:** the walkthrough prepares test states such as |1> and |−>;
        it does not yet provide a generic unknown input |ψ> as in some textbook diagrams.
        `q0` is the physical wire name, while `|ψ>` denotes a state carried by a wire.
        The q# selector injects a test fault only. The correction is chosen from measured
        syndrome bits, not directly from that selector.

        The deterministic walkthrough shows one selected physical error. The statistical
        experiment uses a Qiskit Aer `NoiseModel` attached to data-qubit `id` gates. This
        is a deliberately simplified data-only Pauli-noise model: encoding, syndrome
        extraction, correction, ancillas and measurement readout are ideal. Errors are
        introduced inside the simulator rather than by editing measurement results.
        """
    )
else:
    st.divider()
    st.info(
        f"Running {shots:,} shots per statistical test with a physical error probability "
        f"of {error_pct}%."
    )
    st.caption(
        "This is a syndrome-decoding demonstration, not a machine-learning predictor. "
        "The selected qubit sets the injected teaching fault; the measured syndrome is "
        "what determines the correction."
    )

    try:
        with st.spinner("Building circuits and running Qiskit Aer simulations..."):
            # ------------------------------------------------------------
            # Ideal transmission
            # ------------------------------------------------------------
            if demo == "Ideal transmission":
                qc = build_unprotected("Z", noisy=False)
                show_circuit(qc, "Ideal transmission — no noise", prep_label="|1⟩")

                counts = run_circuit(qc, IDEAL_SHOTS)
                ok = counts.get("1", 0)
                errors = IDEAL_SHOTS - ok

                c1, c2, c3 = st.columns(3)
                c1.metric("Successful transmissions", ok)
                c2.metric("Errors", errors)
                c3.metric("Success rate", f"{100 * ok / IDEAL_SHOTS:.1f}%")

                enc = build_three_qubit_circuit("bitflip", "Z", noisy=False)
                enc_counts = out_strings(enc, run_circuit(enc, IDEAL_SHOTS))
                enc_errors = IDEAL_SHOTS - enc_counts.get("001", 0)
                st.write(
                    f"Extra 3-qubit encoded check: {enc_errors} errors in {IDEAL_SHOTS} shots."
                )

            # ------------------------------------------------------------
            # 3-qubit demos (X, Z, Y)
            # ------------------------------------------------------------
            elif demo in (
                "3-qubit bit-flip code (X)",
                "3-qubit phase-flip code (Z)",
                "Y error — 3-qubit limitation",
            ):
                if demo == "3-qubit bit-flip code (X)":
                    code, pauli, basis = "bitflip", "X", "Z"
                    tests = [("Logical |1>, 0/1 basis", "Z", "Z")]
                    compare_bases = ["Z"]
                elif demo == "3-qubit phase-flip code (Z)":
                    # Match the protected and unprotected input/readout bases.
                    # A Z error is visible on |-> measured in the +/- basis.
                    code, pauli, basis = "phaseflip", "Z", "X"
                    tests = [("Logical |−>, +/- basis", "X", "X")]
                    compare_bases = ["X"]
                else:
                    code, pauli, basis = "bitflip", "Y", "X"
                    tests = [
                        ("Test A: logical |1>, 0/1 basis", "Z", "Z"),
                        ("Test B: logical |->, +/- basis", "X", "X"),
                    ]
                    compare_bases = ["Z", "X"]

                inject = {int(error_qubit): pauli}

                before = build_three_qubit_circuit(code, basis, inject, stage="syndrome")
                show_circuit(
                    before,
                    f"Before correction — {pauli} error on q{error_qubit}",
                    ghost=(pauli, [0, 1, 2]),
                    error_title=ERROR_TITLES[pauli],
                    prep_label=PREP_LABELS[basis],
                    footnote=_footnote(code, pauli),
                )

                syndromes = extract_syndrome(before)
                value = syndromes["syn"]
                s0, s1 = value & 1, (value >> 1) & 1
                found = SYNDROME_TO_INDEX.get(value)

                st.subheader("Measured syndrome")
                st.metric("Syndrome (s₀ s₁)", f"{s0}{s1}")
                st.write("Syndrome table: `00 → no error`, `10 → q0`, `11 → q1`, `01 → q2`.")
                st.caption(
                    "The selected q# is used only to inject the demonstration fault. "
                    "The correction circuit uses the measured syndrome bits to choose its gate."
                )

                if found is not None:
                    correction_gate = "X" if code == "bitflip" else "Z"
                    st.success(
                        f"Syndrome identifies physical qubit q{found}. "
                        f"The code applies {correction_gate} correction on q{found}."
                    )
                else:
                    st.info("The syndrome indicates that no correctable single error was detected.")

                after = build_three_qubit_circuit(code, basis, inject, correction="dynamic")
                show_circuit(
                    after,
                    "After correction, decoding and measurement",
                    ghost=(pauli, [0, 1, 2]),
                    error_title=ERROR_TITLES[pauli],
                    highlight=value,
                    prep_label=PREP_LABELS[basis],
                    footnote=_footnote(code, pauli),
                )

                if demo == "Y error — 3-qubit limitation":
                    st.warning(
                        "The 3-qubit bit-flip code corrects the X component of Y, but it does not "
                        "detect the remaining Z phase component. Test B in the +/- basis makes that "
                        "limitation visible."
                    )

                walkthrough_comparison_table(code, inject, found, compare_bases, 3)

                stats_error = "Y" if pauli == "Y" else pauli
                res = run_statistics(code, stats_error, p, tests, shots=shots)
                show_statistics(res, f"Statistical results — {pauli} noise")

            # ------------------------------------------------------------
            # Shor 9-qubit code
            # ------------------------------------------------------------
            else:
                etype = shor_noise
                walk_type = "Y" if etype == "G" else etype
                inject = {int(error_qubit): walk_type}

                if etype == "G":
                    st.info(
                        "General noise means each noisy `id` gate applies X, Y or Z with equal "
                        "conditional probability. The deterministic walkthrough uses Y as a "
                        "representative single-Pauli case; the statistical run uses the full "
                        "X/Y/Z channel."
                    )

                before = build_shor_circuit("X", inject, stage="syndrome")
                show_circuit(
                    before,
                    f"Shor code before correction — {walk_type} error on q{error_qubit}",
                    error_title=ERROR_TITLES[walk_type],
                    prep_label="|-⟩",
                )

                syndromes = extract_syndrome(before)
                fixes = identify_shor_errors(syndromes)

                st.subheader("Measured syndromes")
                cols = st.columns(4)
                for k in range(3):
                    v = syndromes[f"sx{k}"]
                    cols[k].metric(f"Block {k} bit syndrome", f"{v & 1}{(v >> 1) & 1}")
                v = syndromes["sz"]
                cols[3].metric("Phase syndrome", f"{v & 1}{(v >> 1) & 1}")

                if fixes:
                    st.success(
                        "Detected correction(s): "
                        + ", ".join(f"{gate.upper()} on q{qb}" for gate, qb in fixes)
                    )
                else:
                    st.info("No correctable single-qubit error was detected.")

                # Important: this second circuit performs syndrome measurement and then
                # chooses corrections from its own fresh syndrome bits. It does not receive
                # the selected error_qubit or the pre-run `fixes` list as correction input.
                after = build_shor_circuit("X", inject, correction="dynamic")
                show_circuit(
                    after,
                    "Shor code with syndrome-controlled correction, decoding and measurement",
                    error_title=ERROR_TITLES[walk_type],
                    highlight=syndromes,
                    prep_label="|-⟩",
                )
                st.caption(
                    "The green conditional correction branches correspond to the syndrome "
                    "measured in the separate walkthrough above. The displayed full circuit "
                    "re-measures its own syndrome and controls correction from those fresh bits."
                )

                walkthrough_comparison_table("shor", inject, fixes, ["Z", "X"], 9)

                tests = [
                    ("Test A: logical |1>, 0/1 basis", "Z", "Z"),
                    ("Test B: logical |->, +/- basis", "X", "X"),
                ]
                res = run_statistics("shor", etype, p, tests, shots=shots)
                show_statistics(res, f"Statistical results — {etype} noise")

                st.caption(
                    "With ideal gates and a single-qubit Pauli error, the Shor code can restore "
                    "the logical state. At higher physical error rates, multiple simultaneous "
                    "physical errors become more common and can outweigh the protection."
                )

    except Exception as exc:
        st.error(f"Simulation failed: {type(exc).__name__}: {exc}")
        with st.expander("Technical error details"):
            st.exception(exc)
