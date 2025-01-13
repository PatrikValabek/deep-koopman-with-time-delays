# DeReK: Deep Recursive Koopman for Nonlinear System Identification

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

## Code TODOS

* @PV add deps to requirements.txt (your fork of neuromancer, etc.)
* @PV ensure runability of the code

## Paper TODOS

* see multiple "%%MKL" in tex source
* ChatGPT feeling is strong, e.g. word "Leveraging"
* @PV Abstract is too general, try to rewrite to be more specific to the theory and results from the paper. Basically, good abstract summarizes key points of the paper and provide some sort of "marketing" value
* @PV abbreviations in keywords make no sense
* @PV Rename "Methodology" to content-specific chapter name
* @PV we are missing a formal, mathematical definitions of the Koopman based model and its use in simulation
* @PV Aviod terms like "unnecessary huge matrix", what is a huge matrix?
* @PV Rename "Results" to something like "Case Study", "Numerical Examples" etc. Avoid "generaly speaking section titles"
* Table I: unusable, see comments in tex
* @PV General comment: the paper is missing clear mathematical definitions of key vectors, matrices in equations.
* @PV Na zaklade tohoto modelu sme dostali data, ktore vznikli simulaciou o dobu 100 samples

## Future recommendations

* general suggested labeling in LaTeX: `env_identificator:chapter_label:section_label:actual_label:sub_actual_label`
  * in conference papers: `env_identificator:section_label:actual_label`
  * in subequations/align: `eq:section_label:actual_label` in main environment and then `eq:section_label:actual_label:label_of_individual_eq`

## Acknowledgments

The authors acknowledge the contribution of the Program to support young researchers at STU under the projects AIOperator: eXplainable Intelligence for Industry (AXII) and Koopman Operator in Process Control. The authors gratefully acknowledge the contribution of the Scientific Grant Agency of the Slovak Republic under the grants VEGA 1/0490/23 and 1/0239/24, the Slovak Research and Development Agency under the project APVV--21--0019. This research is funded by the Horizon Europe under the grant no. 101079342 (Fostering Opportunities Towards Slovak Excellence in Advanced Control for Smart Industries).

## Contact

For any questions or issues, please contact Patrik Valábek at [patrik.valabek@stuba.sk](mailto:patrik.valabek@stuba.sk).
