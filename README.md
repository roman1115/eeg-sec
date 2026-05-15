# Learning Cross-Modal Emotional Representations for EEG-Conditioned Speech Emotion Conversion

Official PyTorch implementation of EEG-conditioned speech emotion conversion.

[![Demo](https://img.shields.io/badge/Demo-Online-green)](https://roman1115.github.io/eeg-sec-demo/)
[![Python](https://img.shields.io/badge/Python-3.10-blue)]()

---

## Overview

Speech emotion conversion aims to transform the emotional expression of source speech into a target emotion. This project explores the feasibility of incorporating EEG-derived emotional representations as conditioning inputs for speech emotion conversion through cross-modal alignment.

The framework consists of:

- Speech emotion conversion backbone
- EEG emotion encoder
- Cross-modal representation alignment
- EEG-conditioned speech generation

---

## Framework

<p align="center">
  <img src="assets/fig1.framework.png" width="900"/>
</p>

Overall framework of the proposed EEG-conditioned speech emotion conversion system.  
(a) Training stage, including speech-side pretraining, EEG–speech representation alignment, and EEG-conditioned joint optimization.  
(b) Inference stage with EEG-conditioned speech emotion conversion.

---

## Demo

🌐 Demo Page:

https://roman1115.github.io/eeg-sec-demo/

Audio samples are available on the demo page.
