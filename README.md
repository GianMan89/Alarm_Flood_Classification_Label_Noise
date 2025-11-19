AFC-LaRA: Autoencoder-Based Robustness Analysis for Alarm Flood Classification under Label Noise

This repository contains the code, data, and results accompanying the paper:
“Autoencoder-Based Robustness Analysis for Alarm Flood Classification under Label Noise”.
The work introduces **AFC–LaRA**, a classifier-agnostic framework that uses convolutional
autoencoders to build latent-space label perturbations and evaluate the robustness of
Alarm Flood Classification (AFC) methods to structured label noise.
--------------------------------------------------------------------
Repository Structure
--------------------------------------------------------------------
- classifiers/
  - Implementations and wrappers for the AFC methods evaluated in the paper:
    - WDI_1NN  (Weighted Dissimilarity Index with 1-Nearest Neighbor)
    - EAC_1NN  (Exponentially Attenuated Components + 1-NN)
    - MBW_LR   (Modified Bag-of-Words + Logistic Regression)
    - ACM_SVM  (Alarm Coactivation Matrix + SVM)
    - CASIM    (Convolutional Kernel-based Alarm Subsequence Identification Method)
- data/
  - fcc/
    - Preprocessed alarm-flood CSV files for the Fluidized Catalytic Cracking (FCC) case study.
  - tep/
    - Preprocessed alarm-flood CSV files for the Tennessee–Eastman Process (TEP) case study.
- results/
  - 0_20_25/
    - Intermediate robustness results for 0–20% label noise (2.5% steps).
  - 0_100_5/
    - Intermediate robustness results for 0–100% label noise (5% steps).
  - ae_architecture.pdf
    - Autoencoder architecture figure used in the paper.
  - fcc_ae_architecture.png
    - Autoencoder architecture visualization for the FCC dataset.
  - fcc_robustness_plot_subfigure.{pdf,svg}
    - Robustness plots (global + zoomed inset) for the FCC case study.
  - tep_ae_architecture.png
    - Autoencoder architecture visualization for the TEP dataset.
  - tep_robustness_plot_subfigure.{pdf,svg}
    - Robustness plots (global + zoomed inset) for the TEP case study.
  - results.pdf
    - Combined figures used in the paper.

- notebook.ipynb
  - Example Jupyter notebook demonstrating:
    - Loading the datasets.
    - Training the convolutional autoencoder.
    - Building latent-space label perturbation maps (AFC–LaRA).
    - Running robustness experiments for all AFC methods.

- requirements.txt
  - Python dependencies required to reproduce the experiments.

- LICENSE
  - License information for this repository.

- README.md
  - This documentation file.

--------------------------------------------------------------------
Setup and Usage
--------------------------------------------------------------------

1. Clone the repository:
   git clone https://github.com/GianMan89/Alarm_Flood_Classification_Label_Noise.git

2. Create and activate a Python virtual environment (recommended), then install dependencies:
   pip install -r requirements.txt

3. To reproduce the main experiments and figures, open and run:
   notebook.ipynb

   The notebook will:
   - Load the TEP and FCC alarm-flood datasets from data/.
   - Train the autoencoder and report reconstruction performance.
   - Construct latent-space label perturbation maps (AFC–LaRA).
   - Train and evaluate the AFC classifiers under increasing label noise.
   - Generate the robustness figures stored under results/.

--------------------------------------------------------------------
Data and Code Availability
--------------------------------------------------------------------

All datasets, source code, and experiment outputs used in the paper are included in this repository.
For details on the methodology and experimental results, please refer to the paper:

“Autoencoder-Based Robustness Analysis for Alarm Flood Classification under Label Noise”
(A. Najafi, G. Manca, N. Tamascelli, F. C. Kunze, M. Dix, M. Hollender, A. Fay, T. Chen),
submitted to the **IFAC World Congress 2026** (Busan, Korea).

If you use this repository in your research, please cite the paper accordingly.