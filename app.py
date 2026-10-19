#!/usr/bin/env python3
"""
Quantum Error Correction (QEC) - Interactive Qiskit Demonstration
==================================================================
Demonstrates:
  * 3-qubit bit-flip code   (X errors)
  * 3-qubit phase-flip code (Z errors)
  * Y errors and the limits of the 3-qubit code
  * Shor 9-qubit code       (X, Z, Y / general single-qubit errors)

Install:   pip install qiskit qiskit-aer matplotlib pylatexenc numpy
Run locally:  python -m streamlit run app.py

How noise is simulated
----------------------
Every data qubit passes through an identity gate ("id") that marks the noisy
channel. A Qiskit NoiseModel (pauli_error) attaches the error probability to
that gate, so errors happen INSIDE the quantum simulation. Nothing is ever
changed in the measurement results by hand.
"""

import shutil
import sys
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error

try:
    import matplotlib
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:  # pragma: no cover
    HAVE_MPL = False

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
IDEAL_SHOTS = 100        # transmissions for the ideal (no-noise) baseline
STAT_SHOTS = 2000        # shots per noisy statistical experiment
CHECK_SHOTS = 100        # shots for the deterministic (single-error) walkthroughs
TEXT_WIDTH = max(60, shutil.get_terminal_size((100, 20)).columns - 4)   # text-diagram width

# Syndrome value (s1 s0 as an integer) -> index of the faulty qubit / block.
#   s0 = parity(q0,q1)   s1 = parity(q1,q2)
#   s0=1,s1=0 -> q0 ;  s0=1,s1=1 -> q1 ;  s0=0,s1=1 -> q2
SYNDROME_TO_INDEX = {1: 0, 3: 1, 2: 2}

_diagram_counter = [0]


# ----------------------------------------------------------------------------
# Small console helpers
# ----------------------------------------------------------------------------
def banner():
    print("\n╔════════════════════════════════════╗")
    print("║   QUANTUM ERROR CORRECTION DEMO    ║")
    print("╚════════════════════════════════════╝")


def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def pause(msg="Press Enter to continue..."):
    input(f"\n{msg}")


def state_label(basis):
    return "|1>  (read in the 0/1 basis)" if basis == "Z" else "|->  (read in the +/- basis)"


# ----------------------------------------------------------------------------
# Noise models  (errors happen inside the circuit, on 'id' gates)
# ----------------------------------------------------------------------------
def _noise_model(error):
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(error, ["id"])
    return nm


def create_x_noise_model(p):
    """Bit-flip channel: X with probability p."""
    return _noise_model(pauli_error([("X", p), ("I", 1 - p)]))


def create_z_noise_model(p):
    """Phase-flip channel: Z with probability p."""
    return _noise_model(pauli_error([("Z", p), ("I", 1 - p)]))


def create_y_noise_model(p):
    """Y error (Y = iXZ) with probability p."""
    return _noise_model(pauli_error([("Y", p), ("I", 1 - p)]))


def create_general_noise_model(p):
    """General Pauli noise: with probability p, X, Y or Z (each p/3)."""
    return _noise_model(pauli_error([("X", p / 3), ("Y", p / 3), ("Z", p / 3), ("I", 1 - p)]))


NOISE_FACTORIES = {"X": create_x_noise_model, "Z": create_z_noise_model,
                   "Y": create_y_noise_model, "G": create_general_noise_model}


# ----------------------------------------------------------------------------
# Running circuits
# ----------------------------------------------------------------------------
def run_circuit(qc, shots, noise_model=None):
    """Run on Aer. The stabilizer method is used when possible (all gates are
    Clifford + Pauli noise) so even the 17-qubit Shor circuit is fast."""
    last = None
    for method in ("stabilizer", "automatic"):
        try:
            sim = AerSimulator(method=method, noise_model=noise_model)
            tqc = transpile(qc, sim, optimization_level=0)
            return sim.run(tqc, shots=shots).result().get_counts()
        except Exception as exc:  # fall back to the next method
            last = exc
    raise last


def parse_key(qc, key):
    """Split a counts key into {register_name: bitstring}."""
    names = [cr.name for cr in qc.cregs]
    return dict(zip(reversed(names), key.split()))


def extract_syndrome(qc, shots=CHECK_SHOTS):
    """Run a syndrome circuit and return {register: integer}."""
    counts = run_circuit(qc, shots)
    key = max(counts, key=counts.get)
    regs = parse_key(qc, key)
    return {n: int(v, 2) for n, v in regs.items() if n != "out"}


def out_strings(qc, counts):
    """Counter of final 'out' bitstrings (q_n ... q0)."""
    c = Counter()
    for key, n in counts.items():
        c[parse_key(qc, key)["out"]] += n
    return c


# ----------------------------------------------------------------------------
# Circuit drawing
# ----------------------------------------------------------------------------
DIAGRAM_COLORS = {                      # used by the Qiskit fallback drawer
    "xerror": ("#D62728", "#FFFFFF"), "yerror": ("#D62728", "#FFFFFF"), "zerror": ("#D62728", "#FFFFFF"),
    "xfix": ("#2CA02C", "#FFFFFF"), "zfix": ("#2CA02C", "#FFFFFF"),
}


def draw_circuit(qc, title, scale=0.9, **textbook_opts):
    """Textbook-style diagram (default) -> Qiskit Matplotlib drawing -> plain text.
    The PNG is always saved when graphics work."""
    print(f"\n>>> {title}")
    print("    Legend: red = error (grey dashed = other possible error spots), green = correction,")
    print("            dashed ovals = stages, |psi_k> = state after each stage")
    if HAVE_MPL:
        for mode in ("textbook", "qiskit"):
            try:
                if mode == "textbook":
                    fig = draw_textbook_circuit(qc, title, **textbook_opts)
                else:
                    fig = qc.draw(output="mpl", fold=-1, scale=scale, style={"displaycolor": DIAGRAM_COLORS})
                    fig.suptitle(title, fontsize=13, fontweight="bold")
                _diagram_counter[0] += 1
                fname = f"qec_diagram_{_diagram_counter[0]:02d}.png"
                fig.savefig(fname, dpi=150, bbox_inches="tight")
                print(f"    (diagram saved to {fname})")
                if matplotlib.get_backend().lower() != "agg":      # a display is available
                    print("    (close the diagram window to continue)")
                    plt.show()
                plt.close(fig)
                return
            except Exception as exc:
                print(f"    ({mode} drawing failed: {exc.__class__.__name__}: {exc})")
        print("    Tip: run  pip install pylatexenc matplotlib  for proper diagrams. Using text instead.")
    print(qc.draw(output="text", fold=TEXT_WIDTH, vertical_compression="high"))


# ----------------------------------------------------------------------------
# Textbook-style circuit diagrams (dashed ovals, |psi_k> markers, classically
# controlled corrections). The picture is generated from the SAME Qiskit circuit
# that is simulated, so the diagram always matches what actually runs.
# ----------------------------------------------------------------------------
_GROUP_STYLE = {                           # group key -> (oval label, colour)
    "ENCODE": ("Encode", "#2E8B57"), "DECODE": ("Decode", "#2E8B57"),
    "ERROR": (None, "#D62728"), "CORRECT": ("Correct Error", "#1F5FBF"),
    "MEASURE": ("Measure", "#7B4FA3"),
}
_GROUP_OF = {"SYNDROME": "CORRECT", "X-SYNDROME": "CORRECT", "Z-SYNDROME": "CORRECT",
             "CORRECT": "CORRECT", "ENCODE": "ENCODE", "DECODE": "DECODE",
             "ERROR": "ERROR", "MEASURE": "MEASURE"}
_GAP = 0.9                                 # extra horizontal space between sections


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
    highlight : syndrome value whose controlled correction is drawn as the one that fires
    """
    import matplotlib.pyplot as plt
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

    # wires and labels
    for w in range(n):
        ax.plot([xmin - 0.9, xmax + 0.9], [-w, -w], color="black", lw=1.2, zorder=1)
        reg, idx = qc.find_bit(qc.qubits[w]).registers[0]
        ax.text(xmin - 1.05, -w, f"{reg.name}{idx}  |0⟩", ha="right", va="center", fontsize=11)
        if reg.name in ("q",):
            ax.text(xmax + 1.05, -w, "|ψ⟩" if idx == 0 else "|0⟩", ha="left", va="center", fontsize=11)

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
        if wmax - wmin > 6:                           # many wires (Shor): rounded box instead of a huge oval
            a = (max(gx) - min(gx)) / 2 + 0.6
            b = (wmax - wmin) / 2 + 0.6
            ax.add_patch(FancyBboxPatch((cx - a, cy - b), 2 * a, 2 * b, boxstyle="round,pad=0,rounding_size=0.6",
                                        fill=False, ls="--", ec=color, lw=1.4, zorder=0))
        else:
            a = ((max(gx) - min(gx)) / 2 + 0.55) * 1.32
            b = ((wmax - wmin) / 2 + 0.55) * 1.32
            ax.add_patch(Ellipse((cx, cy), 2 * a, 2 * b, fill=False, ls="--", ec=color, lw=1.4, zorder=0))
        ly = max(cy + b + 0.12, 0.85)                # label never sits on a wire
        ax.text(cx, ly, label, color=color, ha="center", va="bottom", fontsize=11, fontweight="bold")
        top = max(top, ly + 0.7)
        bottom = min(bottom, cy - b - 0.2)

    # section boundaries with |psi_k> markers (counted from after ENCODE)
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
    unit = 0.6                                        # inches per data unit
    w_in = max(9.0, (right - left) * unit)
    h_in = (top + 0.3 - bottom) * unit + 1.5
    fig.set_size_inches(w_in, h_in)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.7 / h_in, top=1 - 0.8 / h_in)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1 - 0.12 / h_in)
    if footnote:
        fig.text(0.5, 0.12 / h_in, footnote, ha="center", va="bottom", fontsize=10, style="italic", color="#333333")
    return fig

# ----------------------------------------------------------------------------
# Building blocks: logical state, error injection
# ----------------------------------------------------------------------------
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
    """A Pauli gate with its own name so it can be coloured in the diagram.
    kind = 'ERROR' (noise we inject) or 'FIX' (correction we apply)."""
    sub = QuantumCircuit(1, name=f"{pauli.lower()}{kind.lower()}")
    getattr(sub, pauli.lower())(0)
    return sub.to_gate(label=f"{pauli} {kind}")


def apply_fix(qc, gate, qubit):
    """Draw/apply an explicit correction gate (green in the diagrams)."""
    qc.append(_marker_gate(gate.upper(), "FIX"), [qubit])


def insert_error(qc, data, inject=None, noisy=False):
    """Error location. `inject` = {qubit_index: 'X'|'Y'|'Z'} places a visible
    error gate (for the walkthrough). `noisy` places 'id' gates that the
    NoiseModel turns into random errors (for statistics)."""
    qc.barrier(label="ERROR")
    for idx, pauli in (inject or {}).items():
        qc.append(_marker_gate(pauli, "ERROR"), [data[idx]])
    if noisy:
        for qb in data:
            qc.id(qb)


# ----------------------------------------------------------------------------
# 3-qubit codes
# ----------------------------------------------------------------------------
def encode_three_qubit_bit_flip(qc, q):
    """|psi> -> |psi psi psi> (in the 0/1 basis) using two CNOTs."""
    qc.cx(q[0], q[1])
    qc.cx(q[0], q[2])


def encode_three_qubit_phase_flip(qc, q):
    """Same as bit-flip encoding, then Hadamards (work in the +/- basis)."""
    encode_three_qubit_bit_flip(qc, q)
    qc.h(q)


def measure_x_syndrome(qc, q, anc, syn):
    """Syndrome for X (bit-flip) errors. anc0 = parity(q0,q1), anc1 = parity(q1,q2)."""
    qc.cx(q[0], anc[0]); qc.cx(q[1], anc[0])
    qc.cx(q[1], anc[1]); qc.cx(q[2], anc[1])
    qc.measure(anc[0], syn[0]); qc.measure(anc[1], syn[1])


def measure_z_syndrome(qc, q, anc, syn):
    """Syndrome for Z (phase-flip) errors. Each ancilla starts in |+> (Hadamard) and is
    the CONTROL of CNOTs onto two neighbouring qubits, which compares their phases
    (an X-type parity check). A Z error on a qubit flips the ancilla (phase kickback),
    so the data qubits are never touched. anc0 checks (q0,q1), anc1 checks (q1,q2),
    giving the same syndrome table as the bit-flip code."""
    for a, (i, j) in zip(anc, [(0, 1), (1, 2)]):
        qc.h(a)
        qc.cx(a, q[i]); qc.cx(a, q[j])
        qc.h(a)
    qc.measure(anc[0], syn[0]); qc.measure(anc[1], syn[1])


def _correct_repetition(qc, q, syn, gate, detected):
    if detected is None:                       # dynamic: driven by the syndrome bits
        for value, idx in SYNDROME_TO_INDEX.items():
            with qc.if_test((syn, value)):
                getattr(qc, gate)(q[idx])
    else:                                      # explicit gate for the identified qubit
        apply_fix(qc, gate, q[detected])


def correct_x_error(qc, q, syn, detected=None):
    """Apply X to the qubit named by the syndrome (never X on every qubit)."""
    _correct_repetition(qc, q, syn, "x", detected)


def correct_z_error(qc, q, syn, detected=None):
    """Apply Z to the qubit named by the syndrome (never Z on every qubit)."""
    _correct_repetition(qc, q, syn, "z", detected)


def decode_three_qubit(qc, q, code):
    """Undo the encoding (phase-flip code: Hadamards first, then the CNOTs)."""
    if code == "phaseflip":
        qc.h(q)
    qc.cx(q[0], q[2])
    qc.cx(q[0], q[1])


def build_three_qubit_circuit(code, basis, inject=None, noisy=False,
                              stage="full", correction="dynamic", fixed=None):
    """
    code      : 'bitflip' or 'phaseflip'
    stage     : 'syndrome' (stop after syndrome measurement) or 'full'
    correction: 'dynamic' (if_test on the syndrome), 'fixed' (explicit gate), 'none'
    """
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


# ----------------------------------------------------------------------------
# Shor 9-qubit code
# ----------------------------------------------------------------------------
def encode_shor(qc, q):
    """Outer phase-flip code across blocks, inner bit-flip code inside blocks."""
    qc.cx(q[0], q[3]); qc.cx(q[0], q[6])
    qc.h([q[0], q[3], q[6]])
    for b in (0, 3, 6):
        qc.cx(q[b], q[b + 1]); qc.cx(q[b], q[b + 2])


def shor_measure_x_syndrome(qc, q, anc, sx):
    """Bit-flip syndromes: two parity checks inside each 3-qubit block."""
    for k in range(3):
        b = 3 * k
        qc.cx(q[b], anc[2 * k]);       qc.cx(q[b + 1], anc[2 * k])
        qc.cx(q[b + 1], anc[2 * k + 1]); qc.cx(q[b + 2], anc[2 * k + 1])
        qc.measure(anc[2 * k], sx[k][0]); qc.measure(anc[2 * k + 1], sx[k][1])


def shor_measure_z_syndrome(qc, q, anc, sz):
    """Phase-flip syndromes: compare the signs of block 0 vs 1 and block 1 vs 2
    (ancilla in |+>, controlled-X onto all 6 qubits of the two blocks)."""
    for i, (b1, b2) in enumerate([(0, 3), (3, 6)]):
        qc.h(anc[i])
        for j in list(range(b1, b1 + 3)) + list(range(b2, b2 + 3)):
            qc.cx(anc[i], q[j])
        qc.h(anc[i])
        qc.measure(anc[i], sz[i])


def decode_shor(qc, q):
    """Exact reverse of the encoder. Afterwards q0 holds the logical qubit."""
    for b in (6, 3, 0):
        qc.cx(q[b], q[b + 2]); qc.cx(q[b], q[b + 1])
    qc.h([q[0], q[3], q[6]])
    qc.cx(q[0], q[6]); qc.cx(q[0], q[3])


def identify_shor_errors(syndromes):
    """Turn measured syndromes into a list of correction gates [(gate, qubit)]."""
    fixes = []
    for k in range(3):
        v = syndromes.get(f"sx{k}", 0)
        if v:
            fixes.append(("x", 3 * k + SYNDROME_TO_INDEX[v]))
    v = syndromes.get("sz", 0)
    if v:
        fixes.append(("z", 3 * SYNDROME_TO_INDEX[v]))     # any qubit of the block works
    return fixes


def build_shor_circuit(basis, inject=None, noisy=False, stage="full",
                       correction="dynamic", fixed=None):
    q = QuantumRegister(9, "q")
    ax = QuantumRegister(6, "ax")          # ancillas for bit-flip checks
    az = QuantumRegister(2, "az")          # ancillas for phase-flip checks
    sx = [ClassicalRegister(2, f"sx{k}") for k in range(3)]
    sz = ClassicalRegister(2, "sz")
    out = ClassicalRegister(9, "out")
    qc = QuantumCircuit(q, ax, az, *sx, sz) if stage == "syndrome" else QuantumCircuit(q, ax, az, *sx, sz, out)

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


# ----------------------------------------------------------------------------
# Unprotected reference qubit
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def get_error_probability():
    while True:
        try:
            raw = input("\nEnter error probability (%) [e.g. 1, 5, 10, 20, 30]: ").strip()
            pct = float(raw.replace("%", ""))
            if 0 < pct <= 50:
                return pct / 100.0
            print("  Please enter a value above 0 and up to 50.")
        except ValueError:
            print("  Please enter a number.")


def run_statistics(code, error_type, p, tests, shots=STAT_SHOTS):
    """
    code       : 'bitflip' | 'phaseflip' | 'shor'
    error_type : 'X' | 'Z' | 'Y' | 'G'
    tests      : list of (label, protected_basis, raw_basis)
    Returns a dict with measured counts.
    """
    nm = NOISE_FACTORIES[error_type](p)
    results = []
    for label, pbasis, rbasis in tests:
        # 1) unprotected qubit through the same noisy channel
        raw_qc = build_unprotected(rbasis)
        raw_counts = run_circuit(raw_qc, shots, nm)
        raw_errors = sum(n for k, n in raw_counts.items() if k != "1")

        # 2) encoded qubit with syndrome measurement + dynamic correction
        if code == "shor":
            qc = build_shor_circuit(pbasis, noisy=True)
        else:
            qc = build_three_qubit_circuit(code, pbasis, noisy=True)
        counts = run_circuit(qc, shots, nm)
        detected = corrected = remaining = 0
        for key, n in counts.items():
            regs = parse_key(qc, key)
            logical_ok = regs["out"][-1] == "1"        # q0 is the right-most bit
            flagged = any(int(v, 2) != 0 for name, v in regs.items() if name != "out")
            if flagged:
                detected += n
                if logical_ok:
                    corrected += n
            if not logical_ok:
                remaining += n
        results.append(dict(label=label, raw_errors=raw_errors, detected=detected,
                            corrected=corrected, remaining=remaining))
    return dict(title=None, p=p, shots=shots, tests=results)


def display_results(title, res):
    shots, p = res["shots"], res["p"]
    line = "-" * 56
    print(f"\n{line}\n{title}\n{line}")
    print(f"Noise probability: {p * 100:g}%")
    print(f"Total shots: {shots} (per test)")
    for t in res["tests"]:
        raw_rate = 100 * t["raw_errors"] / shots
        post_rate = 100 * t["remaining"] / shots
        success = 100 * t["corrected"] / t["detected"] if t["detected"] else float("nan")
        print(f"\n[{t['label']}]")
        print("  RAW ERROR RATE (no QEC, single unprotected qubit):")
        print(f"    Errors: {t['raw_errors']}    Raw error rate: {raw_rate:.2f}%")
        print("  ERROR RATE AFTER QEC (encode, syndrome, correct, decode):")
        print(f"    Shots where the syndrome flagged an error: {t['detected']}")
        print(f"    Successfully corrected: {t['corrected']}")
        print(f"    Remaining (incorrectly decoded): {t['remaining']}")
        print(f"    Post-QEC error rate: {post_rate:.2f}%")
        if t["detected"]:
            print(f"  Correction success rate: {success:.2f}%  (flagged shots that ended correct)")
        if post_rate < raw_rate:
            factor = f"{raw_rate / post_rate:.1f}x lower" if post_rate > 0 else "eliminated"
            print(f"  => QEC REDUCED the error rate: {raw_rate:.2f}% -> {post_rate:.2f}% ({factor})")
        elif post_rate > raw_rate:
            print(f"  => QEC did NOT help here: {raw_rate:.2f}% -> {post_rate:.2f}% (worse)")
        else:
            print(f"  => No difference ({raw_rate:.2f}% both).")
    print(line)


# ----------------------------------------------------------------------------
# Walkthrough helpers (single, visible error)
# ----------------------------------------------------------------------------
def choose_qubit(n, what="physical qubit"):
    while True:
        raw = input(f"\nWhich {what} gets the demo error? (0-{n - 1}, Enter = random): ").strip()
        if raw == "":
            return int(np.random.randint(n))
        if raw.isdigit() and 0 <= int(raw) < n:
            return int(raw)
        print("  Invalid choice.")


ERROR_TITLES = {"X": "Bit-flip Error (X)", "Z": "Phase Error (Z)", "Y": "Y Error (X and Z)"}
PREP_LABELS = {"Z": "|1⟩", "X": "|−⟩"}


def _footnote(code, pauli=None):
    if pauli == "Y":
        return ("Y = iXZ: the bit-flip code corrects only the X part. The Z part is NOT detected, "
                "so a phase error remains after this correction.")
    if code == "phaseflip":
        return ("Each ancilla (H, controlled-X, H) compares the phase of two neighbouring qubits; a Z error flips it. "
                "The syndrome (s0 s1) selects which controlled Z fires (green = the one that fired).")
    return ("Dashed boxes = possible error spots. The syndrome (s0 s1) selects which controlled X fires "
            "(green = the one that fired).")


def print_syndrome_table():
    print("\nSyndrome table  (s0 s1):  s0 = parity(q0,q1),  s1 = parity(q1,q2)")
    print("   00 -> no error      10 -> q0      11 -> q1      01 -> q2")


def compare_before_after(builder, bases, n_bits):
    """Run the same single error without and with correction, show the results."""
    expected = "0" * (n_bits - 1) + "1"
    for basis in bases:
        print(f"\n  Logical state {state_label(basis)}   expected output: {expected}")
        for name, kwargs in (("Without correction", dict(correction="none")),
                             ("With correction   ", dict(correction="fixed"))):
            qc = builder(basis, **kwargs)
            counts = out_strings(qc, run_circuit(qc, CHECK_SHOTS))
            bits, n = counts.most_common(1)[0]
            ok = bits == expected
            if ok:
                verdict = "logical state restored"
            elif bits[-1] == "1":
                verdict = "logical bit OK, but other qubits still corrupted"
            else:
                verdict = "LOGICAL ERROR"
            print(f"    {name}: {bits}  ({100 * n // CHECK_SHOTS}%)  -> {verdict}")


# ----------------------------------------------------------------------------
# Demonstrations
# ----------------------------------------------------------------------------
def ideal_simulation():
    section("IDEAL CONDITION - NO NOISE  (baseline)")
    print("Baseline: a qubit is sent 100 times with NO noise of any kind.")
    print("No bit flips, no phase flips, no Y errors are introduced.\n")

    qc = build_unprotected("Z", noisy=False)
    draw_circuit(qc, "Ideal transmission (no noise anywhere)")
    counts = run_circuit(qc, IDEAL_SHOTS)          # no noise model at all
    ok = counts.get("1", 0)
    err = IDEAL_SHOTS - ok

    # same check with the full 3-qubit QEC machinery but still no noise
    enc = build_three_qubit_circuit("bitflip", "Z", noisy=False)
    enc_counts = out_strings(enc, run_circuit(enc, IDEAL_SHOTS))
    enc_err = IDEAL_SHOTS - enc_counts.get("001", 0)

    print("\n" + "-" * 44)
    print("IDEAL (BASELINE) RESULT")
    print("-" * 44)
    print(f"Total transmissions       = {IDEAL_SHOTS}")
    print(f"Errors                    = {err}")
    print(f"Error percentage          = {100 * err / IDEAL_SHOTS:.0f}%")
    print(f"Successful transmissions  = {ok}")
    print(f"Success percentage        = {100 * ok / IDEAL_SHOTS:.0f}%")
    print("-" * 44)
    print(f"Extra check: encoded 3-qubit circuit with no noise -> {enc_err} errors in {IDEAL_SHOTS}")
    print("(The encoding/syndrome/decoding steps add no errors by themselves.)")
    print("\nThis is the reference: any errors in later demos come from the noise we add.")


def _three_qubit_walkthrough(code, pauli, diagram_basis, compare_bases):
    """Shared steps 3-7 for the X, Z and Y demos. Returns the identified qubit."""
    gate_name = {"bitflip": "X", "phaseflip": "Z"}[code]
    err_q = choose_qubit(3)
    inject = {err_q: pauli}

    print(f"\nStep 1-3: prepare logical qubit, encode into 3 qubits, inject {pauli} error on q{err_q}")
    before = build_three_qubit_circuit(code, diagram_basis, inject, stage="syndrome")
    draw_circuit(before, f"DIAGRAM 1 - Before correction: encoded system + {pauli} error on q{err_q} + syndrome measurement",
                 ghost=(pauli, [0, 1, 2]), error_title=ERROR_TITLES[pauli],
                 prep_label=PREP_LABELS[diagram_basis], footnote=_footnote(code, pauli))

    print("\nStep 4-5: measure the syndrome with ancilla qubits and identify the error")
    syndromes = extract_syndrome(before)
    value = syndromes["syn"]
    s0, s1 = value & 1, (value >> 1) & 1
    found = SYNDROME_TO_INDEX.get(value)
    print_syndrome_table()
    print("\nSYNDROME RESULT")
    print("---------------")
    print(f"Syndrome: {s0}{s1}")
    if found is None:
        print("Interpretation: no error detected.")
        return None
    print(f"\nInterpretation:\n  The syndrome points to physical qubit q{found}.")
    print(f"Correction:\n  {gate_name} correction on q{found}  (chosen from the syndrome, not applied to every qubit).")

    after = build_three_qubit_circuit(code, diagram_basis, inject, correction="dynamic")
    draw_circuit(after, f"DIAGRAM 2 - After correction: syndrome -> detected error on q{found} -> {gate_name}-correction on q{found} -> decode -> measure",
                 ghost=(pauli, [0, 1, 2]), error_title=ERROR_TITLES[pauli], highlight=value,
                 prep_label=PREP_LABELS[diagram_basis], footnote=_footnote(code, pauli))

    print("\nStep 6-7: compare the final result without and with correction")
    builder = lambda basis, **kw: build_three_qubit_circuit(code, basis, inject, fixed=found, **kw)
    compare_before_after(builder, compare_bases, 3)
    return found


def three_qubit_x_demo():
    section("3-QUBIT BIT-FLIP CODE - X ERROR")
    print("Idea: |psi> -> |psi psi psi>   (encode with two CNOT gates)")
    print("An X error flips ONE qubit. Parity checks on ancilla qubits reveal WHICH one")
    print("without measuring the data itself, then we flip it back.\n")
    print("X error -> bit flip -> syndrome identifies affected qubit -> X correction applied.")
    p = get_error_probability()
    found = _three_qubit_walkthrough("bitflip", "X", "Z", ["Z"])

    print("\nRunning statistical simulation with Qiskit NoiseModel ...")
    res = run_statistics("bitflip", "X", p, [("Logical |1>, 0/1 basis", "Z", "Z")])
    display_results("X ERROR SIMULATION (3-qubit bit-flip code)", res)
    print("\nWhy it works: one flipped qubit is out-voted by the other two.")
    print("It fails only when 2+ qubits flip, which is rare: about 3p^2 instead of p.")


def three_qubit_z_demo():
    section("3-QUBIT PHASE-FLIP CODE - Z ERROR")
    print("A Z error changes the PHASE of a qubit (|+> <-> |->), not its 0/1 value,")
    print("so measuring in the 0/1 basis cannot see it directly.")
    print("Idea: encode in the +/- basis (CNOTs + Hadamards). Two ancilla qubits compare the")
    print("PHASE of neighbouring qubits (the ancilla flips if a Z error is in between), so the")
    print("syndrome points to the qubit with the Z error, and a Z gate fixes it.\n")
    print("Z error -> syndrome extraction -> identify affected qubit -> Z correction -> measure.")
    p = get_error_probability()
    _three_qubit_walkthrough("phaseflip", "Z", "Z", ["Z"])

    print("\nRunning statistical simulation with Qiskit NoiseModel ...")
    print("(The unprotected reference qubit is measured in the +/- basis, because a")
    print(" Z error is invisible in the 0/1 basis.)")
    res = run_statistics("phaseflip", "Z", p,
                         [("Protected logical |1> vs unprotected |->", "Z", "X")])
    display_results("Z ERROR SIMULATION (3-qubit phase-flip code)", res)
    print("\nSame majority-vote idea as the bit-flip code, but in the Hadamard basis.")


def three_qubit_y_demo(offer_shor=True):
    section("Y ERROR DEMONSTRATION")
    print("Y = iXZ  ->  a Y error is a bit flip (X) AND a phase flip (Z) together.")
    print("It is NOT just another X error.\n")
    print("The 3-qubit BIT-FLIP code only checks for X-type errors. It will detect and")
    print("fix the X part of Y, but the Z part passes through untouched.")
    p = get_error_probability()
    # logical |-> makes the leftover Z visible (logical |1> alone hides it)
    _three_qubit_walkthrough("bitflip", "Y", "X", ["Z", "X"])

    print("\nRunning statistical simulation with Qiskit NoiseModel ...")
    res = run_statistics("bitflip", "Y", p, [
        ("Test A: logical |1>, 0/1 basis", "Z", "Z"),
        ("Test B: logical |->, +/- basis", "X", "X"),
    ])
    display_results("Y ERROR SIMULATION (3-qubit bit-flip code)", res)

    print("\nLIMITATION OF THE 3-QUBIT CODE")
    print("------------------------------")
    print("Test A looks fine because the X part of Y is corrected. Test B exposes the")
    print("leftover Z (phase) error. A repetition code protects against ONE error type")
    print("only. To fix X, Z and therefore Y, we need a code that handles both: the")
    print("Shor 9-qubit code (bit-flip code nested inside a phase-flip code).")
    if offer_shor:
        if input("\nRun the Shor 9-qubit code on a Y error now? (y/n): ").strip().lower() == "y":
            shor_code_demo(preset="Y")


def _choose_shor_error():
    print("\nError type for the Shor code:")
    print("  1. X (bit flip)   2. Z (phase flip)   3. Y (X and Z)   4. General (random X/Y/Z)")
    while True:
        c = input("Choice (1-4): ").strip()
        if c in ("1", "2", "3", "4"):
            return {"1": "X", "2": "Z", "3": "Y", "4": "G"}[c]
        print("  Invalid choice.")


def shor_code_demo(preset=None):
    section("SHOR 9-QUBIT CODE")
    print("Structure: 3 blocks of 3 qubits.")
    print("  * Inside each block: bit-flip code      -> fixes X errors")
    print("  * Across the blocks: phase-flip code    -> fixes Z errors")
    print("  * Y = iXZ is fixed because both parts are found and corrected separately.")
    etype = preset or _choose_shor_error()
    p = get_error_probability()

    walk_type = "Y" if etype == "G" else etype
    err_q = choose_qubit(9)
    inject = {err_q: walk_type}
    print(f"\nStep 1-3: prepare logical qubit, encode with Shor code, inject {walk_type} error on q{err_q}")
    if walk_type == "Y":
        print("          (Y = iXZ: this single gate contains BOTH an X part and a Z part)")

    before = build_shor_circuit("X", inject, stage="syndrome")
    draw_circuit(before, f"DIAGRAM 1 - Shor code before correction: {walk_type} error on q{err_q} + syndrome measurement",
                 error_title=ERROR_TITLES[walk_type], prep_label="|-⟩")

    print("\nStep 4-5: measure syndromes and identify the error")
    syndromes = extract_syndrome(before)
    fixes = identify_shor_errors(syndromes)
    print("\nSYNDROME RESULT")
    print("---------------")
    for k in range(3):
        v = syndromes[f"sx{k}"]
        print(f"Block {k} bit-flip syndrome (s0 s1): {v & 1}{(v >> 1) & 1}")
    v = syndromes["sz"]
    print(f"Between-blocks phase syndrome (s0 s1): {v & 1}{(v >> 1) & 1}")
    print("\nInterpretation / Correction:")
    if not fixes:
        print("  No error detected.")
    for gate, qubit in fixes:
        if gate == "x":
            print(f"  X component found on q{qubit} -> apply X(q{qubit})")
        else:
            blk = qubit // 3
            print(f"  Z component found in block {blk} -> apply Z(q{qubit})  (any qubit of the block works)")
    if walk_type == "Y":
        print("  => Both an X part AND a Z part were found: that is exactly a Y error.")

    after = build_shor_circuit("X", inject, correction="fixed", fixed=fixes)
    draw_circuit(after, "DIAGRAM 2 - Shor code after correction: identified gates applied, decode, measure",
                 error_title=ERROR_TITLES[walk_type], prep_label="|-⟩")

    print("\nStep 6-8: decode and compare without / with correction")
    builder = lambda basis, **kw: build_shor_circuit(basis, inject, fixed=fixes, **kw)
    compare_before_after(builder, ["Z", "X"], 9)

    print("\nRunning statistical simulation with Qiskit NoiseModel (17 qubits, stabilizer simulator) ...")
    tests = [("Test A: logical |1>, 0/1 basis", "Z", "Z"),
             ("Test B: logical |->, +/- basis", "X", "X")]
    res = run_statistics("shor", etype, p, tests)
    names = {"X": "X", "Z": "Z", "Y": "Y", "G": "GENERAL (X/Y/Z)"}
    display_results(f"{names[etype]} ERROR SIMULATION (Shor 9-qubit code)", res)
    worse = any(t["remaining"] > t["raw_errors"] for t in res["tests"])
    print("\nThe Shor code corrects ANY single-qubit X, Z or Y error. It fails only when two or")
    print("more of the 9 physical qubits are hit in the same round.")
    if worse:
        print("At this noise level that happens often: 9 qubits are exposed instead of 1, so the")
        print("code is past its break-even point. This is real behaviour (the 'threshold').")
        print("Try a lower probability such as 1% or 2% to see QEC win.")
    else:
        print("At this noise level two simultaneous errors are rare, so the post-QEC error rate")
        print("falls roughly like p^2 instead of p.")


# ----------------------------------------------------------------------------
# Menu
# ----------------------------------------------------------------------------
def show_menu():
    print("\n========================================")
    print("       QUANTUM ERROR CORRECTION")
    print("========================================")
    print("1. Ideal condition - No noise")
    print("2. 3-Qubit Bit-Flip Error (X)")
    print("3. 3-Qubit Phase-Flip Error (Z)")
    print("4. Y Error Demonstration")
    print("5. Shor 9-Qubit Code - Y/General Error")
    print("6. Exit")
    return input("\nEnter your choice: ").strip()


def main():
    banner()
    actions = {"1": ideal_simulation, "2": three_qubit_x_demo, "3": three_qubit_z_demo,
               "4": three_qubit_y_demo, "5": shor_code_demo}
    while True:
        try:
            choice = show_menu()
            if choice == "6":
                print("\nGoodbye!")
                break
            if choice in actions:
                actions[choice]()
                pause("Press Enter to return to the main menu...")
            else:
                print("Invalid choice, please enter 1-6.")
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting.")
            break

# =============================================================================
# Streamlit interface
# =============================================================================
import streamlit as st

st.set_page_config(
    page_title="Quantum Error Correction Lab",
    page_icon="⚛️",
    layout="wide",
)

st.title("⚛️ Quantum Error Correction Lab")
st.caption(
    "Interactive Qiskit/Aer simulation of the same QEC demonstrations in the original CLI program. "
    "No quantum hardware is used."
)

with st.sidebar:
    st.header("Simulation settings")

    demo = st.selectbox(
        "Choose a demonstration",
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
        min_value=1,
        max_value=50,
        value=5,
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
        min_value=0,
        max_value=max_qubit,
        value=0,
        step=1,
        help=f"Choose q0–q{max_qubit} for the deterministic single-error walkthrough.",
    )

    shor_noise = "Y"
    if demo == "Shor 9-qubit code":
        shor_noise = st.selectbox(
            "Shor-code noise channel",
            ["X", "Z", "Y", "G"],
            index=2,
            format_func=lambda x: {
                "X": "X (bit flip)",
                "Z": "Z (phase flip)",
                "Y": "Y (combined X + Z)",
                "G": "General random X/Y/Z",
            }[x],
        )

    run_button = st.button(
        "Run simulation",
        type="primary",
        width="stretch",
    )

p = error_pct / 100.0


def show_circuit(qc, title, **kwargs):
    st.subheader(title)
    try:
        fig = draw_textbook_circuit(qc, title, **kwargs)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    except Exception as exc:
        st.warning(
            f"Textbook diagram unavailable ({type(exc).__name__}: {exc}). Showing the Qiskit text circuit instead."
        )
        st.code(qc.draw(output="text", fold=120), language="text")


def show_statistics(res, title):
    st.subheader(title)
    rows = []

    for t in res["tests"]:
        raw_rate = 100 * t["raw_errors"] / res["shots"]
        post_rate = 100 * t["remaining"] / res["shots"]
        success = (
            100 * t["corrected"] / t["detected"]
            if t["detected"]
            else 0.0
        )
        rows.append(
            {
                "Test": t["label"],
                "Unprotected errors": t["raw_errors"],
                "Raw error rate (%)": round(raw_rate, 2),
                "Syndrome-flagged shots": t["detected"],
                "Successfully corrected": t["corrected"],
                "Remaining errors": t["remaining"],
                "Post-QEC error rate (%)": round(post_rate, 2),
                "Correction success (%)": round(success, 2),
            }
        )

    st.dataframe(rows, width="stretch", hide_index=True)

    for row in rows:
        raw = row["Raw error rate (%)"]
        post = row["Post-QEC error rate (%)"]
        if post < raw:
            st.success(
                f'{row["Test"]}: QEC reduced the measured error rate from {raw:.2f}% to {post:.2f}%.'
            )
        elif post > raw:
            st.warning(
                f'{row["Test"]}: QEC increased the measured error rate from {raw:.2f}% to {post:.2f}%. '
                "At this physical error rate, the extra exposed qubits can outweigh the protection."
            )
        else:
            st.info(f'{row["Test"]}: no measured change in error rate.')


if not run_button:
    st.markdown(
        """
        ### About this lab

        This interface keeps the original program's quantum-circuit workflow:

        **prepare → encode → inject/simulate noise → measure syndrome → correct → decode → measure**

        - **Ideal transmission:** no noise baseline.
        - **3-qubit bit-flip code:** protects against a single X error.
        - **3-qubit phase-flip code:** protects against a single Z error in the Hadamard basis.
        - **Y-error demonstration:** shows the limitation of using only the bit-flip repetition code.
        - **Shor 9-qubit code:** combines bit-flip and phase-flip protection for arbitrary single-qubit Pauli errors.

        The deterministic walkthrough shows a selected physical error. The statistical experiment uses a
        Qiskit Aer `NoiseModel`, attaching the selected Pauli channel to the circuit's `id` gates, so the
        errors are introduced inside the quantum simulation rather than by editing measurement results.
        """
    )
else:
    st.divider()
    st.info(
        f"Running {shots:,} shots per statistical test with a physical error probability of {error_pct}% ."
    )

    try:
        with st.spinner("Building circuits and running Qiskit Aer simulations..."):
            if demo == "Ideal transmission":
                qc = build_unprotected("Z", noisy=False)
                show_circuit(qc, "Ideal transmission — no noise")

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

            elif demo in (
                "3-qubit bit-flip code (X)",
                "3-qubit phase-flip code (Z)",
                "Y error — 3-qubit limitation",
            ):
                if demo == "3-qubit bit-flip code (X)":
                    code, pauli, basis, tests = (
                        "bitflip",
                        "X",
                        "Z",
                        [("Logical |1>, 0/1 basis", "Z", "Z")],
                    )
                elif demo == "3-qubit phase-flip code (Z)":
                    code, pauli, basis, tests = (
                        "phaseflip",
                        "Z",
                        "Z",
                        [("Protected logical |1> vs unprotected |->", "Z", "X")],
                    )
                else:
                    code, pauli, basis, tests = (
                        "bitflip",
                        "Y",
                        "X",
                        [
                            ("Test A: logical |1>, 0/1 basis", "Z", "Z"),
                            ("Test B: logical |->, +/- basis", "X", "X"),
                        ],
                    )

                inject = {int(error_qubit): pauli}

                before = build_three_qubit_circuit(
                    code,
                    basis,
                    inject,
                    stage="syndrome",
                )
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
                st.write(
                    "Syndrome table: `00 → no error`, `10 → q0`, `11 → q1`, `01 → q2`."
                )

                if found is not None:
                    correction_gate = "X" if code == "bitflip" else "Z"
                    st.success(
                        f"Syndrome identifies physical qubit q{found}. "
                        f"The code applies {correction_gate} correction on q{found}."
                    )
                else:
                    st.info("The syndrome indicates that no correctable single error was detected.")

                after = build_three_qubit_circuit(
                    code,
                    basis,
                    inject,
                    correction="dynamic",
                )
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
                        "The 3-qubit bit-flip code corrects the X component of Y, but it does not detect "
                        "the remaining Z phase component. The +/- basis in Test B makes that limitation visible."
                    )

                stats_error = "Y" if pauli == "Y" else pauli
                res = run_statistics(
                    code,
                    stats_error,
                    p,
                    tests,
                    shots=shots,
                )
                show_statistics(res, f"Statistical results — {pauli} noise")

            else:
                etype = shor_noise
                walk_type = "Y" if etype == "G" else etype
                inject = {int(error_qubit): walk_type}

                if etype == "G":
                    st.info(
                        "General noise means each noisy `id` gate applies X, Y or Z with equal conditional "
                        "probability. The deterministic walkthrough uses Y as a representative single-Pauli case; "
                        "the statistical run uses the full X/Y/Z channel."
                    )

                before = build_shor_circuit(
                    "X",
                    inject,
                    stage="syndrome",
                )
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
                    cols[k].metric(
                        f"Block {k} bit syndrome",
                        f"{v & 1}{(v >> 1) & 1}",
                    )
                v = syndromes["sz"]
                cols[3].metric(
                    "Phase syndrome",
                    f"{v & 1}{(v >> 1) & 1}",
                )

                if fixes:
                    st.success(
                        "Detected correction(s): "
                        + ", ".join(f"{gate.upper()} on q{qb}" for gate, qb in fixes)
                    )
                else:
                    st.info("No correctable single-qubit error was detected.")

                after = build_shor_circuit(
                    "X",
                    inject,
                    correction="fixed",
                    fixed=fixes,
                )
                show_circuit(
                    after,
                    "Shor code after correction, decoding and measurement",
                    error_title=ERROR_TITLES[walk_type],
                    prep_label="|-⟩",
                )

                tests = [
                    ("Test A: logical |1>, 0/1 basis", "Z", "Z"),
                    ("Test B: logical |->, +/- basis", "X", "X"),
                ]
                res = run_statistics(
                    "shor",
                    etype,
                    p,
                    tests,
                    shots=shots,
                )
                show_statistics(res, f"Statistical results — {etype} noise")

                st.caption(
                    "With ideal gates and a single-qubit Pauli error, the Shor code can restore the logical state. "
                    "At higher physical error rates, multiple simultaneous physical errors become more common and "
                    "can outweigh the protection."
                )

    except Exception as exc:
        st.error(f"Simulation failed: {type(exc).__name__}: {exc}")
        with st.expander("Technical error details"):
            st.exception(exc)
