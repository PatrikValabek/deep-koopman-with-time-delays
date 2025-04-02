# DeReK: Deep Recursive Koopman for Nonlinear System Identification

## Paper TODOS

* Chyba mi v clanku definicia simulacnych scenarov. Co keby chcel niekto zreprodukovat simulacny experiment? Ma vsetky podklady k tomu?

## Code TODOS

* @PV add deps to requirements.txt (your fork of neuromancer, etc.)
* @PV ensure runability of the code

## Overview

This repository contains the LaTeX source files for the publication titled **"Deep Dictionary-Free Method for Identifying Linear Model of Nonlinear System with Input Delay"**. The paper discusses the application of the Koopman theory and LSTM in system identification, providing mathematical definitions, methodologies, and case studies.

## Repository Structure

```plaintext
code/
examples/
functions/
requirements.txt
results/
document/
├── figures/
├── main.bib
└── root.tex
README.md
```

## Acknowledgments

The authors acknowledge the contribution of the Program to support young researchers at STU under the projects AIOperator: eXplainable Intelligence for Industry (AXII) and Koopman Operator in Process Control. The authors gratefully acknowledge the contribution of the Scientific Grant Agency of the Slovak Republic under the grants VEGA 1/0490/23 and 1/0239/24, the Slovak Research and Development Agency under the project APVV--21--0019. This research is funded by the Horizon Europe under the grant no. 101079342 (Fostering Opportunities Towards Slovak Excellence in Advanced Control for Smart Industries).

## Contact

For any questions or issues, please contact Patrik Valábek at [patrik.valabek@stuba.sk](mailto:patrik.valabek@stuba.sk).

# 📋 PC25 Revision Checklist

## ✅ 1st Review

### 🔍 Technical & Conceptual Comments

- **PV1**: ✅ Clarify the definition and position of `x_k` in Section 2.A.  
  - 🔲 Fix Fig. 1 showing `x_{k-1}` as last input.  
  - 🔲 Formally define the sequence `H`.  
  _Difficulty: Medium_

- **PV2**: ✅ Include `h_k` in function `g` throughout Section 2.B.  
  - 🔲 Revise equations (3)–(6) to reflect:  
    `x_k - g^{-1}(g(x_k,h_k))` and  
    `h_k = f(x_k, ..., x_{k-nH}, u_k, ..., u_{k-nH})`  
  _Difficulty: Important conceptual fix_

- **MW3**: Improve clarity of **Figure 3**.  
  - 🔲 Explain if it shows eigenvalues of linearized system.  
  - 🔲 Clarify cross colors in lower subplots.

- **PV4**: ✅ Add brief explanation on solving optimization problem (7).  
  - 🔲 Include weight values used.

- **PV5**: ✅ Define and justify the value of the sequence length `H`.  
  - 🔲 Was `H = 20` used to match system time delay?

- **PV6**: ✅ Explain testing initialization.  
  - 🔲 Was ground truth sequence used for the first LSTM prediction?

- **MW7**: Improve comparison methodology.  
  - 🔲 Compare Figure 2 results with ground truth (simulated from Eq. 8), not noisy data.  
  - 🔲 Add metric vs. ground truth.  

- **MW8**: Evaluation metrics clarification.  
  - 🔲 Show **SAD** in Table 1 (not just **MAE**). or delete sad from text :D

- **PV9**: ✅ Fix cross-references.  
  - 🔲 Correct output trajectory loss reference (end of page 3). 

## ✅ 2nd Review

### 🌀 Clarification & Explanation Needed

- **MW**: Further explain **eigenvalue analysis in Fig. 3**.  
  - 🔲 Clarify what the “original” eigenvalues refer to.  
  - 🔲 Explain color and size coding in the complex plane.

### 📝 Grammatical & Style Suggestions

- **MW1**: In Abstract, introduce **eDMD** like LSTM.

- **MW4**: Reword “we aim to find matrices in state equation covering …” — currently unclear.

- **MW5**: In Equation (2), verify upper sum limit `n`.  
  - 🔲 Is it related to `z_k` dimension or number of snapshots (`m`)?

- **PV6**:?ask MK? Fix grammar:  
  - 🔲 “These loss functions are described in the [11].” → missing article or rephrase.

- **PV8**: ✅Clarify Lifting Network explanation.  
  - 🔲 Sentence: “two layers, 60 neurons, resulting in 40 states” — split and explain transformation.

- **MW9**: Fix table reference:  
  - ✅ “Table III-A” incorrect.  
  - 🔲 Shorten the table caption.  
  - 🔲 Add **SAD** metric (if missing).

### 🌟 Optional Suggestions

- **MW11 (Optional)**:  
  - 🔲 Add noise-free ground truth to **Fig. 2**.  
  - 🔲 Include detail view on time steps where **input flow rate changes** to show **time delay**.

- **MW12**:  
  - 🔲 Explain eigenvalue color/size coding in Fig. 3.  
  - 💡 Option: Make all markers same color/size to simplify and avoid the need for detailed explanation.
