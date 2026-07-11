# 🔍 XAI-Q Audit Framework for Intrusion Detection Systems
> *Practical Explainability Auditing for Resource-Constrained Security Operations Centers*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Hardware](https://img.shields.io/badge/Hardware-4GB_GPU%20%26%20CPU--Friendly-orange)]()

## 📌 Overview
This repository implements the **XAI-Q Audit Framework**, a lightweight, reproducible methodology for evaluating the practical utility of explainable AI (XAI) in Intrusion Detection Systems (IDS). 

Designed for resource-constrained environments (≤4GB GPU, CPU-only training), this work addresses a critical gap in cybersecurity research: **most XAI studies optimize accuracy but ignore explanation stability, sparsity, and operational actionability for SOC analysts.**

### 🔑 Core Contributions
1. **XAI-Q Score**: Composite metric combining Explanation Stability (ESS), Sparsity Index (SI), and Actionability Rating (AR)
2. **Deployment Checklist**: 10-point operational rubric for SOC readiness assessment
3. **CPU-Optimized Pipeline**: Fully reproducible on standard laptops using `scikit-learn`, `SHAP`, and `LIME`
4. **Zero-Budget Design**: Uses public datasets, open-source tools, and free-tier infrastructure

## ⚙️ Hardware & Software Requirements
| Component | Specification |
|-----------|---------------|
| **CPU** | Intel i5 / Ryzen 5 or equivalent (4+ cores) |
| **GPU** | 4GB NVIDIA (optional; pipeline runs on CPU) |
| **RAM** | 8GB+ recommended (16GB ideal) |
| **OS** | Windows 10/11, Ubuntu 20.04+, macOS |
| **Python** | 3.8–3.11 |

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/[your-username]/xai-q-audit-ids.git
cd xai-q-audit-ids
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
