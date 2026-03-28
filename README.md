# On the Use of Large Language Models as Judges for Mathematics Teaching Quality Assessment

This repository contains the code accompanying the Master's thesis *"On the Use of Large Language Models as Judges for Mathematics Teaching Quality Assessment"* in **Cognitive Neuroscience and Clinical Neuropsychology** at the University of Padua (UniPD), conducted under the supervision of Prof. Alberto Testolin and the co-supervision of Prof. Christiaan de Kock.

The project focuses on the use of large language models as judges within the Mathematical Quality of Instruction (MQI) framework.

This repository is shared primarily to document and illustrate the work carried out for the thesis. It should not be considered a fully cleaned or production-ready codebase. Some notebooks still contain drafts, exploratory analyses, and trial-and-error steps.

## Overview

This repository contains code to:

- prepare MQI datasets for analysis,
- reconstruct lesson- or segment-level textual units from classroom transcripts,
- align transcript segments with human ratings,
- compare LLM ratings with human ratings,
- evaluate prompt configurations and model variants,
- analyze agreement at both segment/lesson and teacher levels.

## Repository structure

- `prepare_dataset.py`  
  End-to-end dataset preparation pipeline. Loads rating files and transcript units, cleans them, pivots rater columns, and merges ratings with textual data.

- `concatenate_utt_per_chapter.py`  
  Builds lesson- or segment-level text units from utterance-level transcripts. Chapters are approximated from word-count windows so that utterances can be assigned to rating segments.

- `create_examples.py`  
  Helper script for constructing example material used in prompting workflows.

- `Find_best_configuration.ipynb`  
  Compares prompt configurations and model variants to identify the optimal setup.

- `Kripp_human.ipynb`  
  Computes human agreement baselines for the MQI dataset.

- `Kripp_balanced-subsets_*.ipynb`  
  Contains agreement analyses on selected balanced subsets and test MQI dimensions.

- `Kripp_teachers-subsets_2.5-flash_best-config.ipynb`  
  Contains teacher-level subset analyses.

- `LLM-as-Judges_on_MQI/ICPSR_36095/2 Mathematical Quality of Instruction/MQI 4-Point.pdf`  
  MQI rating codebook taken from [NCTE Classroom Transcript Analysis](https://github.com/ddemszky/classroom-transcript-analysis), the GitHub repository of Demszky and Hill (2023).

- `VertexAI/`  
  Contains the scripts used to run the LLM generations.

- `generations/`  
  Contains the codebooks and examples provided to the LLM for each MQI dimension.

- `data/`  
  Contains the scripts used to construct the different subsets.

- `qualitative_analysis_project/qualitative_analysis/metrics/`  
  Contains metric-related code adapted from [LLM4Humanities](https://github.com/flowersteam/LLM4Humanities), the GitHub repository by Clerc (2025), with some adjustments for this project. It also includes an implementation of Gwet's agreement coefficient, inspired by the Krippendorff-based code but without bootstrapping.