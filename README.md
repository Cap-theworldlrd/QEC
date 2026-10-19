<div align="center">

# ⚛️ Quantum Error Correction Lab

### An Interactive Qiskit/Aer Simulation of Repetition & Shor Codes

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.x-6929C4?logo=qiskit&logoColor=white)](https://qiskit.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*A pedagogical quantum error correction simulator that shows — step by step — how
bit-flip, phase-flip, and Shor 9-qubit codes detect and correct physical errors,
with the noise injected **inside** the quantum circuit.*

</div>

---

## Table of Contents

1. [Overview](#1-overview)
2. [Physics Background](#2-physics-background)
3. [The Codes — Theory](#3-the-codes--theory)
   - 3.1 [3-Qubit Bit-Flip Code](#31-3-qubit-bit-flip-code)
   - 3.2 [3-Qubit Phase-Flip Code](#32-3-qubit-phase-flip-code)
   - 3.3 [Why Y Breaks the Repetition Code](#33-why-y-breaks-the-repetition-code)
   - 3.4 [Shor 9-Qubit Code](#34-shor-9-qubit-code)
4. [Noise Model & Simulation Methodology](#4-noise-model--simulation-methodology)
5. [How to Use the App](#5-how-to-use-the-app)
6. [Technical Architecture (Code)](#6-technical-architecture-code)
7. [Mathematical Appendix](#7-mathematical-appendix)
8. [Interpreting the Results](#8-interpreting-the-results)
9. [Limitations & Future Work](#9-limitations--future-work)
10. [Installation](#10-installation)
11. [References](#11-references)
12. [License](#12-license)

---

## 1. Overview

Quantum computers promise exponential speed-ups for certain problems, but
their fundamental unit — the **qubit** — is fragile. Any coupling to the
environment, imperfect gate, or thermal fluctuation can corrupt the state
before a computation finishes. Classical repetition codes are not directly
usable because the **no-cloning theorem** forbids copying an unknown quantum
state. Yet quantum error correction (QEC) *is* possible: by encoding one
logical qubit into several entangled physical qubits, we can detect and undo
errors without ever measuring — and thus collapsing — the logical state.

This project is a **from-scratch, transparent simulation** of three
landmark QEC schemes:

| Code | Physical qubits | Corrects | Introduced by |
|------|-----------------|----------|---------------|
| 3-qubit bit-flip | 3 | X errors | (folklore, 1995) |
| 3-qubit phase-flip | 3 | Z errors | (folklore, 1995) |
| Shor 9-qubit | 9 | any single-qubit X, Y, Z | Peter Shor (1995) |

Everything is built with **Qiskit**, executed on the **Aer** simulator, and
visualised through a **Streamlit** front-end. Noise is injected with a
`NoiseModel` attached to identity gates, so the errors happen **inside the
quantum simulation** — never by post-hoc editing of measurement outcomes.

The interactive app shows, for every demonstration:

1. The circuit **before** correction (with a chosen error injected).
2. The **measured syndrome** and the qubit it identifies.
3. The circuit **after** correction (with the fired correction highlighted).
4. A **deterministic** side-by-side comparison: without vs. with correction.
5. A **statistical** experiment (thousands of shots) comparing the raw error
   rate with the post-QEC error rate.

---

## 2. Physics Background

### 2.1 The Pauli error basis

A single-qubit quantum state is $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$.
Any single-qubit error can be expanded in the **Pauli basis**:

$$
E = e_I I + e_X X + e_Y Y + e_Z Z,
\qquad
X = \begin{pmatrix}0 & 1\\ 1 & 0\end{pmatrix},\;
Z = \begin{pmatrix}1 & 0\\ 0 & -1\end{pmatrix},\;
Y = \begin{pmatrix}0 & -i\\ i & 0\end{pmatrix}.
$$

The three Pauli operators have intuitive physical meanings:

| Error | Action on basis states | Interpretation |
|-------|------------------------|----------------|
| $X$ | $\|0\rangle \leftrightarrow \|1\rangle$ | **Bit flip** |
| $Z$ | $\|+\rangle \leftrightarrow \|-\rangle$ | **Phase flip** |
| $Y = iXZ$ | Both | **Bit + phase flip** |

Because Pauli matrices form a basis, correcting $X$, $Z$, and $Y = iXZ$
errors is sufficient to correct **any** single-qubit error (a general
unitary error is a linear combination, and QEC corrects each basis term
independently — this is the **discretisation of quantum errors**, a
cornerstone result).

### 2.2 The Pauli noise channel

In this simulator, every data qubit passes through a **Pauli channel** on an
identity gate:

$$
\mathcal{E}(\rho) = (1-p)\,\rho \;+\; p_X X\rho X \;+\; p_Y Y\rho Y \;+\; p_Z Z\rho Z,
\qquad
p_X + p_Y + p_Z = p,
$$

where $p$ is the physical error probability set by the sidebar slider. Four
channels are available:

| Channel | Definition | Used in |
|---------|-----------|---------|
| `X` | $p_X = p$ only | Bit-flip demo |
| `Z` | $p_Z = p$ only | Phase-flip demo |
| `Y` | $p_Y = p$ only | Y-error demo |
| `G` (general) | $p_X = p_Y = p_Z = p/3$ | Shor general demo |

### 2.3 Why QEC is non-trivial

Three obstacles make naive error correction impossible:

1. **No-cloning.** Unknown quantum states cannot be copied; the encoder must
   entangle rather than copy.
2. **Measurement collapses.** We cannot inspect a data qubit to see whether
   it flipped.
3. **Errors are continuous.** An error is an arbitrary unitary, not a finite
   alphabet.

QEC solves all three by **encoding** the logical qubit into an entangled
subspace and by measuring only **stabiliser generators** — commuting,
multi-qubit observables whose eigenvalues (the *syndrome*) reveal the error
type without leaking any information about $\alpha$ or $\beta$.

---

## 3. The Codes — Theory

Throughout this section $|\psi\rangle_L = \alpha|0\rangle_L + \beta|1\rangle_L$
denotes the logical qubit, and $\mathcal{C}$ denotes the code subspace.

### 3.1 3-Qubit Bit-Flip Code

**Encoding.** Map the logical state to a repetition code across three
physical qubits:

$$
|\psi\rangle_L = \alpha|0\rangle + \beta|1\rangle
\;\xrightarrow{\;\text{CNOT}_{0\to1},\,\text{CNOT}_{0\to2}\;}
\alpha|000\rangle + \beta|111\rangle.
$$

The two stabilisers of the code are

$$
S_0 = Z_0 Z_1, \qquad S_1 = Z_1 Z_2,
$$

with $S_i|\psi\rangle_L = +|\psi\rangle_L$. Any **single** $X$ error
anticommutes with exactly two of the $Z$ checks (one, for the edge qubits),
producing a measurable $-1$ eigenvalue.

**Syndrome extraction.** Two ancilla qubits are prepared in $|0\rangle$ and
used as targets of CNOTs:

$$
a_0 = q_0 \oplus q_1 \pmod 2, \qquad a_1 = q_1 \oplus q_2 \pmod 2.
$$

Measuring $a_0, a_1$ in the computational basis gives a two-bit **syndrome**
$s = (s_0, s_1)$, yielding this error table:

| $s_0$ | $s_1$ | Diagnosis | Correction |
|:-----:|:-----:|-----------|:----------:|
| 0 | 0 | no error | — |
| 1 | 0 | $X_0$ | $X_0$ |
| 1 | 1 | $X_1$ | $X_1$ |
| 0 | 1 | $X_2$ | $X_2$ |

The syndrome is encoded as an integer $s_0 + 2 s_1$ in code
(`SYNDROME_TO_INDEX = {1: 0, 3: 1, 2: 2}`).

**Correction.** A classically-controlled $X$ is applied to the qubit named
by the syndrome:

$$
\text{if } s = (1,0)\colon X_0;\quad s = (1,1)\colon X_1;\quad s = (0,1)\colon X_2.
$$

**Logical error rate.** The code fails iff **two or more** physical qubits
flip:

$$
P_L^{(X)} = 3p^2(1-p) + p^3 = 3p^2 - 2p^3 \;\approx\; 3p^2 \text{ for } p \ll 1.
$$

So if $p = 5\%$, the raw failure probability is $5\%$ but the encoded
failure probability is only $\approx 0.75\%$ — a $\sim 6.7\times$
improvement. Break-even (where $P_L < p$) occurs at $3p^2 - 2p^3 < p$,
i.e. roughly $p < 1/3$; below that, **more repetition always helps**.

---

### 3.2 3-Qubit Phase-Flip Code

Phase errors $Z$ are invisible in the computational basis. The solution is
to work in the **Hadamard basis** $\{|+\rangle, |-\rangle\}$, where a $Z$
error acts like a bit flip.

**Encoding.** Repeat the bit-flip encoding and then Hadamard every qubit:

$$
|\psi\rangle_L = \alpha|0\rangle + \beta|1\rangle
\;\xrightarrow{\;\text{encode}\;}
\alpha|000\rangle + \beta|111\rangle
\;\xrightarrow{\;H^{\otimes 3}\;}
\alpha|{+}{+}{+}\rangle + \beta|{-}{-}{-}\rangle.
$$

Equivalently, in the $|+\rangle,|-\rangle$ basis the logical information is
repetition-coded, and a $Z$ error on one physical qubit is now a bit flip.

**Stabilisers.** With respect to the code subspace, the stabilisers become
X-type:

$$
S_0 = X_0 X_1, \qquad S_1 = X_1 X_2.
$$

**Syndrome extraction.** Each ancilla is prepared in $|+\rangle$, used as
the **control** of two CNOTs onto neighbouring data qubits, then Hadamard-ed
back:

$$
\underbrace{H\,(\text{CNOT}_{a_0 \to q_0})(\text{CNOT}_{a_0 \to q_1})\,H}_{a_0},
\qquad
\underbrace{H\,(\text{CNOT}_{a_1 \to q_1})(\text{CNOT}_{a_1 \to q_2})\,H}_{a_1}.
$$

If a $Z$ error hits $q_i$, the corresponding ancilla flips $|+\rangle
\leftrightarrow |-\rangle$ — a **phase kickback** that is read out in the
computational basis. The data qubits are never touched.

**Correction.** A classically-controlled $Z$ is applied to the identified
qubit.

**Logical error rate.** Identical to §3.1: $P_L^{(Z)} \approx 3p^2$.

---

### 3.3 Why Y Breaks the Repetition Code

The Pauli $Y$ operator decomposes as

$$
Y = i X Z.
$$

A $Y$ error is therefore a *combined* bit flip **and** phase flip. The
bit-flip repetition code has stabilisers $Z_0 Z_1, Z_1 Z_2$; these commute
with the $Z$-part of $Y$ but anticommute with the $X$-part. Consequently:

- The **$X$-part is detected and corrected** — the syndrome flags the
  qubit, and a physical $X$ is applied.
- The **$Z$-part is invisible** to the bit-flip syndrome — no $X_0 X_1$
  check exists — and survives in the decoded logical qubit as a logical
  $Z_L$ error.

This is exactly what the Y-error demonstration exposes:

| Test | Basis | Expected if no $Z$ survival | Observed |
|------|-------|-----------------------------|----------|
| A | $Z$ (0/1) | Logical qubit OK | ✅ pass |
| B | $X$ ($\pm$) | Logical qubit OK | ❌ fail — logical $Z_L$ |

**Conclusion.** A repetition code protects against **one type** of error
only. To correct $X$, $Z$ **and** $Y$, we need a code whose stabiliser group
contains both $Z$-type and $X$-type checks — the Shor code.

---

### 3.4 Shor 9-Qubit Code

Peter Shor's 1995 construction concatenates the phase-flip code (outer)
with the bit-flip code (inner) to build the first code that corrects an
**arbitrary single-qubit error**.

**Encoding.** Start with one logical qubit $q_0$. Apply:

1. **Outer phase-flip code:** two CNOTs $q_0 \to q_3, q_0 \to q_6$ spread
   the state across three **blocks** of qubits.
2. **Hadamards** on $q_0, q_3, q_6$ switch to the $+/-$ basis for the outer
   code.
3. **Inner bit-flip code:** inside each block $b \in \{0, 3, 6\}$, apply
   CNOTs $q_b \to q_{b+1}$ and $q_b \to q_{b+2}$.

Result:

$$
|\psi\rangle_L \;\xrightarrow{\text{encode}}\;
\alpha \bigl(|000\rangle+|111\rangle\bigr)^{\otimes 3}
\;+\;
\beta \bigl(|000\rangle-|111\rangle\bigr)^{\otimes 3}.
$$

**Stabilisers (8 total).**

*Bit-flip checks (6):* two per block —
$Z_b Z_{b+1}$ and $Z_{b+1} Z_{b+2}$ for $b \in \{0, 3, 6\}$.

*Phase-flip checks (2):* pairwise $X$ products across blocks —
$X_0 X_1 X_2 X_3 X_4 X_5$ and $X_3 X_4 X_5 X_6 X_7 X_8$.

Because these 8 generators are independent and commute, the code space is
$2^{9-8} = 2$ dimensional — i.e., **one logical qubit**.

**Syndrome extraction.**

- **X syndromes (`sx0`, `sx1`, `sx2`):** Each block $k$ uses two ancilla
  qubits prepared in $|0\rangle$, driven by CNOTs from $q_{3k}, q_{3k+1},
  q_{3k+2}$ exactly like the 3-qubit bit-flip code. A single $X$ in block
  $k$ yields a non-zero syndrome local to that block.
- **Z syndromes (`sz`):** Two ancilla qubits prepared in $|+\rangle$ act as
  controls on all six qubits of two adjacent blocks (block 0 vs 1, and block
  1 vs 2). A $Z$ on *any* of the 9 qubits flips exactly one of these two
  checks, which identifies *which block* the phase error is in (but not
  which qubit inside the block — that's fine, since a logical $Z_L$ acts
  identically on any qubit of the block).

**Correction.**

- For each non-zero `sx_k`, apply $X$ to the qubit identified by the
  3-qubit syndrome table inside block $k$.
- For a non-zero `sz`, apply $Z$ to the first qubit of the identified block
  (any qubit of the block gives the same logical action).

**Why this corrects Y.** A $Y$ error triggers **both** an $X$-syndrome
(inside its block) **and** a $Z$-syndrome (across blocks). Both corrections
are applied, and the combined action $X_i Z_j$ on physically distinct
qubits reproduces — up to an irrelevant global phase — the inverse of $Y$.

**Logical error rate.** The Shor code fails iff at least two physical
errors occur in the same round. Ignoring combinatorial factors of order 1,

$$
P_L^{\text{Shor}} \;\sim\; \binom{9}{2} p^2 = 36\,p^2.
$$

For small $p$ this is far worse than the single-qubit raw error rate — the
**overhead of 9 exposed qubits must be beaten by the $p^2$ scaling**. This
defines the **break-even** or **threshold**:

$$
P_L(p) < p \quad \Longleftrightarrow \quad 36\,p^2 < p \quad
\Longleftrightarrow \quad p < \tfrac{1}{36} \approx 2.8\%.
$$

Below $\approx 2.8\%$ physical error per qubit per round, the Shor code
**reduces** the logical error rate; above it, the code is past break-even
and **amplifies** the error. The slider in the app goes up to $50\%$ precisely
so you can **see the threshold**.

> **Note.** The $\sim 36 p^2$ formula is a crude first-order estimate; the
> exact value depends on the fault-tolerance assumptions (ancilla
> preparation, measurement, gate errors). The app runs an *ideal-gate*
> simulation with noise only on data qubits, so the observed crossover
> occurs somewhat higher than $2.8\%$.

---

## 4. Noise Model & Simulation Methodology

The critical design choice in this project is that **noise is physical**.
An earlier, naive approach would have been to simulate the ideal circuit
and then flip bits in the output histogram with probability $p$ — that is
*not* a quantum simulation. Instead, the app:

1. Attaches a `NoiseModel` to every **identity gate** placed on data
   qubits during the "error" section of the circuit:

   ```python
   nm = NoiseModel()
   nm.add_all_qubit_quantum_error(
       pauli_error([("X", p), ("I", 1 - p)]), ["id"]
   )
   ```

2. Inserts `qc.id(qb)` on every data qubit right after the encoded state
   is prepared. The simulator's **density-matrix / stabilizer propagation**
   then samples each `id` gate and applies the Pauli error *inside* the
   state, before the syndrome measurement.

Consequences:

- The syndrome measurement **sees** the error because the error is
  genuinely present in the quantum state.
- Multi-qubit errors arise from **independent** single-qubit errors on
  different `id` gates — exactly the physical scenario that QEC must
  overcome.
- Corrections are applied based on the **measured** syndrome, not on the
  injected error — the two agree for a single error, but disagree whenever
  ≥ 2 physical qubits flip, which is the failure mode QEC must handle.

### 4.1 Simulator backend

`run_circuit()` tries backends in this order:

1. **`stabilizer`** — Clifford circuits (all gates used here are Clifford)
   with Pauli noise. It scales to the 17-qubit Shor circuit (9 data + 6 `x`-
   ancilla + 2 `z`-ancilla) and runs 2000 shots in well under a second.
2. **`automatic`** — fallback that picks the best method per circuit.

---

## 5. How to Use the App

### Sidebar

| Control | Effect |
|---------|--------|
| **Demonstration** | Select ideal baseline, 3-qubit codes, or Shor code. |
| **Physical error probability (%)** | Sets $p$ for the statistical run only; the deterministic walkthrough uses a fixed single injected error. |
| **Shots per statistical test** | Samples per statistical experiment (100 – 5000). |
| **Qubit receiving the walkthrough error** | The qubit $q_i$ on which the walkthrough injects the error. |
| **Shor-code noise channel** | $X$, $Z$, $Y$, or general for the Shor demo. |
| **Run simulation** | Executes the demo. |
| **Reset** | Returns to the overview screen. |

### Results screen

Every demonstration shows:

1. **Before-correction circuit diagram** — the encoded state, the injected
   error (red), the syndrome measurement, and dashed "possible error" boxes
   on all other qubits.
2. **Measured syndrome** — the numeric $(s_0, s_1)$ pair (or six bits for
   Shor) plus its interpretation.
3. **After-correction circuit diagram** — the fired correction is drawn in
   **green**; non-fired corrections remain dashed.
4. **Deterministic comparison table** — same injected error, without vs.
   with correction, in both the $0/1$ and $+/-$ bases.
5. **Statistical table** — raw error rate (one unprotected qubit) versus
   post-QEC error rate (encoded + corrected), for each basis. Green rows
   mean QEC helped; amber rows mean QEC made things worse (past break-even).

---

## 6. Technical Architecture (Code)

### 6.1 File layout

```
qec_app.py
│
├── Configuration
│   ├── IDEAL_SHOTS, STAT_SHOTS, CHECK_SHOTS
│   └── SYNDROME_TO_INDEX, ERROR_TITLES, PREP_LABELS
│
├── Noise models (4 factories)
│   └── create_{x,z,y,general}_noise_model → pauli_error
│
├── Circuit execution
│   ├── run_circuit(qc, shots, noise_model)       # stabilizer → automatic
│   ├── parse_key(qc, key)                        # split counts into regs
│   ├── extract_syndrome(qc)                      # most-frequent syndrome
│   └── out_strings(qc, counts)                   # just the "out" register
│
├── Textbook diagram renderer
│   ├── _layout_circuit(qc)                       # column & section layout
│   ├── draw_textbook_circuit(qc, title, ...)     # matplotlib figure
│   └── _footnote(code, pauli)                    # per-code footer text
│
├── Building blocks
│   ├── prepare_logical / measure_logical
│   ├── _marker_gate / apply_fix                  # coloured error & fix gates
│   └── insert_error(qc, data, inject, noisy)     # ERROR or id gates
│
├── Code builders
│   ├── 3-qubit: encode / syndrome / correct / decode
│   │            build_three_qubit_circuit(...)
│   └── Shor  : encode / sx-syndrome / sz-syndrome / decode / identify
│                build_shor_circuit(...)
│
├── Statistics
│   └── run_statistics(code, error_type, p, tests, shots)
│
└── Streamlit UI
    ├── sidebar (inputs, run/reset buttons)
    ├── show_circuit(qc, title, ...)
    ├── show_statistics(res, title)
    ├── walkthrough_comparison_table(...)
    └── dispatch on demo name
```

### 6.2 Circuit builder contract

Both `build_three_qubit_circuit()` and `build_shor_circuit()` accept the same
keyword arguments, so the UI can dispatch uniformly:

```python
build_three_qubit_circuit(
    code       = "bitflip" | "phaseflip",
    basis      = "Z" | "X",                   # logical state |1> or |->
    inject     = {qubit_index: "X"|"Y"|"Z"},  # walkthrough error (visible)
    noisy      = True | False,                # statistical noise via id gates
    stage      = "syndrome" | "full",
    correction = "dynamic" | "fixed" | "none",
    fixed      = qubit_index | None,          # required if correction="fixed"
)
```

`stage="syndrome"` stops the circuit right after the ancilla measurements,
which is what the *before-correction* diagrams show. `stage="full"` continues
through decode and data measurement.

`correction="dynamic"` uses `qc.if_test((syn, value))` — a real
classically-controlled correction driven by the measured syndrome. This is
what the statistical simulation uses. `correction="fixed"` applies an
explicit `X`/`Z` marker gate to a qubit chosen by the user — this is what
the walkthrough uses so the fired correction can be drawn in green.

### 6.3 Error injection

`insert_error(qc, data, inject=None, noisy=False)` places:

- **A visible `X/Y/Z error` gate** on each qubit in `inject` (walkthrough).
- **An `id` gate** on every data qubit when `noisy=True` (statistics). The
  `NoiseModel` attaches the Pauli channel to those `id` gates, so the error
  occurs *inside* the simulation.

Every error/fix gate is defined by `_marker_gate(pauli, kind)` — a small
named sub-circuit — so the diagram renderer can detect it by name
(`"xerror"`, `"zfix"`, etc.) and colour it appropriately.

### 6.4 Diagram renderer

`draw_textbook_circuit()` reconstructs a hand-drawn-style figure from the
actual `QuantumCircuit` object:

1. `_layout_circuit()` walks `qc.data`, assigns every instruction a column,
   and starts a new **section** at every labelled barrier (`PREPARE`,
   `ENCODE`, `ERROR`, `SYNDROME`, `CORRECT`, `DECODE`, `MEASURE`).
2. Sections are merged into **groups** (`ENCODE`, `DECODE`, `ERROR`,
   `CORRECT`, `MEASURE`) and enclosed in dashed ovals (or rounded boxes
   for tall sections like Shor's 9-qubit block).
3. Vertical dashed lines with $|\psi_k\rangle$ markers are drawn at every
   section boundary after `ENCODE`, giving the reader the state evolution.
4. `if_else` operations render as classically-controlled boxes connected to
   the measurement symbol by a classical wire; **the branch whose value
   matches `highlight` is coloured green** — this is how the diagram shows
   *which* correction fired for the specific error the user injected.

### 6.5 Statistics engine

`run_statistics()` runs, for each test:

1. **Unprotected baseline.** A one-qubit circuit through the same noisy `id`
   gate → raw error rate.
2. **Encoded + corrected.** The 3-qubit or Shor circuit with `noisy=True`
   and `correction="dynamic"` → post-QEC error rate.

Both are compared per *logical state* (0/1 vs $+/-$ basis). Because $q_0$ is
the **right-most** bit of the `out` register, `regs["out"][-1]` cleanly
extracts the logical bit, independent of qubit count (1, 3, or 9).

### 6.6 Session state

The **Run simulation** button sets `st.session_state.show_results = True`.
The **Reset** button clears it. This ensures that adjusting a slider after
a run does not bounce the user back to the overview — a common Streamlit
pitfall.

---

## 7. Mathematical Appendix

### 7.1 Syndrome table derivation (3-qubit code)

Let $q \in \{0,1\}^3$ be the encoded bit-string. Define parities

$$
s_0 = q_0 \oplus q_1, \qquad s_1 = q_1 \oplus q_2.
$$

Acting on $|000\rangle$ with a single $X_i$ gives:

| Error | $q$ | $s_0$ | $s_1$ | Integer |
|:-----:|:---:|:-----:|:-----:|:-------:|
| $I$ | 000 | 0 | 0 | 0 |
| $X_0$ | 100 | 1 | 0 | 1 |
| $X_1$ | 010 | 1 | 1 | 3 |
| $X_2$ | 001 | 0 | 1 | 2 |

The mapping is a bijection from $\{q_0, q_1, q_2\}$ to
$\{1, 2, 3\}$ — the syndrome *is* the qubit index.

### 7.2 Shor code stabiliser group

The stabiliser generators of the Shor code are:

$$
\begin{aligned}
Z_0 Z_1,&\quad Z_1 Z_2 &&\text{(block 0 bit parity)}\\
Z_3 Z_4,&\quad Z_4 Z_5 &&\text{(block 1 bit parity)}\\
Z_6 Z_7,&\quad Z_7 Z_8 &&\text{(block 2 bit parity)}\\
X_0 X_1 X_2 & X_3 X_4 X_5 &&\text{(phase parity block 0 vs 1)}\\
X_3 X_4 X_5 & X_6 X_7 X_8 &&\text{(phase parity block 1 vs 2)}
\end{aligned}
$$

These 8 operators commute and are independent, so the logical code space has
dimension $2^{9-8} = 2$.

### 7.3 Failure probabilities

**3-qubit code.** A logical failure occurs for any $X$ error pattern in which
$\geq 2$ qubits flip:

$$
P_L = \sum_{k=2}^{3} \binom{3}{k} p^k (1-p)^{3-k}
= 3p^2(1-p) + p^3.
$$

**Shor code.** A logical failure occurs for any error pattern with $\geq 2$
affected qubits (from any of the 8 stabilisers being violated):

$$
P_L^{\text{Shor}} \approx \binom{9}{2} p^2 = 36 p^2 + O(p^3).
$$

**Break-even.** The crossover $P_L(p) = p$ yields:

- 3-qubit code: $p^\star \approx 1/3$;
- Shor code: $p^\star \approx 1/36 \approx 2.8\%$.

---

## 8. Interpreting the Results

The statistical table has these columns:

| Column | Meaning |
|--------|---------|
| **Unprotected errors** | Absolute count of logical errors in the raw-qubit run. |
| **Raw error rate (%)** | $100 \times \text{errors} / \text{shots}$. |
| **Syndrome-flagged shots** | Runs in which the syndrome was non-zero. |
| **Successfully corrected** | Flagged runs that decoded to the correct logical bit. |
| **Remaining errors** | Runs that decoded to the wrong logical bit. |
| **Post-QEC error rate (%)** | $100 \times \text{remaining} / \text{shots}$. |
| **Correction success (%)** | $100 \times \text{corrected} / \text{detected}$. |

Colour coding:

- 🟩 **Green row** — post-QEC rate < raw rate. QEC helped.
- 🟨 **Amber row** — post-QEC rate > raw rate. Physical error rate is above
  threshold (for the Shor code, try $p \leq 2\%$ to see QEC win).
- ⬜ **Grey row** — no change.

If you see the Shor code make things *worse* at $p = 20\%$, that is not a
bug — the code has 9 physical qubits exposed versus 1, and the $\sim 36p^2$
term dominates at high $p$. This is the **threshold behaviour** that makes
fault-tolerant quantum computing hard and is exactly the effect the demo
wants to expose.

---

## 9. Limitations & Future Work

- **Ideal gates.** Only the $id$ gates carry noise; the encoding, syndrome,
  and decode gates are perfect. A fully fault-tolerant simulation would
  also model gate errors.
- **Ancilla noise.** The measurement ancillas are error-free; in a real
  device they contribute to the syndrome.
- **Single-round code.** Only one round of syndrome extraction is performed.
  Real fault-tolerant schemes repeat extraction to handle measurement
  errors.
- **No surface code.** The Shor code is a pedagogical example; modern
  hardware favours the surface code (higher threshold, nearest-neighbour
  checks).
- **No error modelling beyond Pauli.** Coherent and leakage errors are not
  simulated (though the discretisation theorem makes Pauli noise a good
  proxy for *depolarising* channels).

Possible extensions:

- Add a **Steane 7-qubit code** and a **surface code** demo.
- Add a **fault-tolerant flag** that also injects noise on CNOTs and
  measurements.
- Add **correlated noise** channels (Kraus operators).
- Add a **logical error rate curve** plot (error rate vs. $p$) to make the
  threshold explicit.

---

## 10. Installation

### Requirements

```
python >= 3.9
qiskit >= 1.0
qiskit-aer >= 0.14
matplotlib >= 3.7
pylatexenc >= 2.10
numpy >= 1.24
streamlit >= 1.30
```

### Install

```bash
pip install qiskit qiskit-aer matplotlib pylatexenc numpy streamlit
```

### Run

```bash
streamlit run qec_app.py
```

Then open <http://localhost:8501> in your browser.

### Quick sanity check

On launch, select **Ideal transmission** and click **Run simulation**. You
should see a **100 %** success rate and no errors — confirming the simulator
is set up correctly. Then try **Shor 9-qubit code** with $p = 2\%$ in the
`Y` channel: the post-QEC error rate should be visibly lower than the raw
error rate.

---

## 11. References

1. P. W. Shor, *Scheme for reducing decoherence in quantum computer memory*,
   Phys. Rev. A **52**, R2493 (1995).
2. A. M. Steane, *Error correcting codes in quantum theory*,
   Phys. Rev. Lett. **77**, 793 (1996).
3. D. Gottesman, *Stabilizer Codes and Quantum Error Correction*, PhD
   thesis, Caltech (1997). arXiv:quant-ph/9705052.
4. M. A. Nielsen and I. L. Chuang, *Quantum Computation and Quantum
   Information*, Cambridge University Press (2010), §10.
5. J. Preskill, *Quantum Computing in the NISQ era and beyond*,
   Quantum **2**, 79 (2018).
6. Qiskit Aer documentation: <https://qiskit.github.io/qiskit-aer/>.

---

## 12. License

Released under the **MIT License** — see [LICENSE](LICENSE) for details.

<div align="center">

**Built with Qiskit + Streamlit · No quantum hardware required**

</div>
