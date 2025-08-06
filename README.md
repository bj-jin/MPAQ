# MPAQ: A Multi-persona Framework for Argument Quality Assessment
## Dataset
Our work is based on the IBM-Rank-30k (Gretz et al., 2020) and IBM-ArgQ-5.3kArgs (Toledo et al., 2019) datasets.
> Shai Gretz, Roni Friedman, Edo Cohen-Karlik, Assaf Toledo, Dan Lahav, Ranit Aharonov, and Noam Slonim. 2020. A large-scale dataset for argument quality ranking: Construction and analysis.

> Assaf Toledo, Shai Gretz, Edo Cohen-Karlik, Roni Friedman, Elad Venezian, Dan Lahav, Michal Jacovi, Ranit Aharonov, and Noam Slonim. 2019. Automatic argument quality assessment - new datasets and methods.
## Persona Generation
Run this command to train the LoRA for Persona Generation:
```bash
bash persona.sh
```
## Quality Assessment
> Before performing this stage, please first copy the persona file of the test split to "./data/{dataset}"
Run this command to train the LoRA for Quality Assessment:
```bash
bash quality_assessment.sh
```
