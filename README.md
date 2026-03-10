# Replication Package for the Paper:  
## *CPArchPSLinker: Cross-Platform Linking of Architectural Solutions from Q&A Platforms to Architectural Problems in Commits and Issues*

This replication package accompanies the paper **“CPArchPSLinker: Cross-Platform Linking of Architectural Solutions from Q&A Platforms to Architectural Problems in Commits and Issues.”**

The repository provides an overview of **CPArchPSLinker**, along with its source code and baseline implementations, the dataset of GitHub commits/issues and Stack Overflow posts used in the study, and the experimental results reported in the paper.

---

## 🚨 Introduction

Collaborative development platforms such as **GitHub** and Q&A websites such as **Stack Overflow (SO)** serve as complementary knowledge sources within the **Open Source Software (OSS)** ecosystem. When developers encounter architectural problems during OSS development, such as architectural anti-patterns, modularization issues, or performance bottlenecks, they often consult SO to identify potential solutions.

However, the unstructured, heterogeneous, and divergent nature of discussions on SO makes identifying relevant architectural solutions time-consuming and labor-intensive. To address this challenge, we define the problem of **linking architectural knowledge across Software Engineering (SE) platforms** (GitHub and SO) and introduce **CPArchPSLinker**, an automated approach designed for this task.

---

## 🏗️ CPArchPSLinker Overview

**CPArchPSLinker** is an approach for automatically linking architectural solutions from Q&A platforms to architectural problems described in GitHub commits and issues. The approach operates in two main stages.

### **Stage 1 – Identification of Relevant ⟨Architectural Problem, Solution⟩ Pairs**

In the first stage, CPArchPSLinker employs a **Deep Metric Learning (DML)**–based model to address cross-platform heterogeneity and distribution divergence between GitHub and SO artifacts. The DML model jointly projects architectural problems described in commits or issues and architectural solutions discussed in SO posts into a shared embedding space.

The model is trained such that semantically relevant ⟨architectural problem, solution⟩ pairs are mapped closer together in this space, while irrelevant pairs are pushed farther apart. This learned metric space enables the identification of cross-platform relevant pairs beyond surface-level textual similarity.

### **Stage 2 – Linking Architectural Problems in Commits/Issues to Solutions on SO**

In the second stage, CPArchPSLinker performs architectural problem–solution linking by ranking candidate solutions from SO for a given architectural problem described in a GitHub commit or issue.

This stage integrates multiple feature types, including  **Lexical features**, **Sentence-BERT embeddings**, **architecture-aware association features**, and the **relevance probabilities predicted by the DML-based model in Stage 1**. These features are jointly leveraged within a **learning-to-rank model** to link each architectural problem to its most relevant solutions and produce a ranked list of candidate solutions.

---

## 🧩 CPArchPSLinker Architecture

The **CPArchPSLinker architecture** consists of five primary layers.

![CPArchPSLinker Architecture](image/CPArchPSLinker_Architecture.png)

## 📁 Repository Structure

```plaintext
├── data                  # Datasets used in the study

├── image/                # Contains a figure that illustrates the architecture of the proposed approach, CPArchPSLinker.

├── models/               # Source code for the CPArchPSLinker and baseline models

├── results/              # Experimental outputs

└── README.md             # Overview and usage instructions for this repository
```

## 🛠️ Dataset Description

The `data/` directory contains the following datasets:

- **`CrossPArchPSBench.xlsx`** – The benchmark dataset proposed in this study for evaluating techniques that link architectural solutions from SO to architectural problems described in GitHub commits and issues.  
  It contains **5,068 labeled ⟨architectural problem, solution⟩ pairs**, including **2,534 relevant (positive)** pairs and **2,534 irrelevant (negative)** pairs. Each pair links a GitHub architectural problem with a SO architectural solution and is annotated according to whether the solution addresses the problem. This benchmark dataset provides a reusable resource for future research, enabling consistent evaluation and comparison of architectural problem–solution linking approaches.

- **`2,071_Commits_Issues.xlsx`** – Contains **2,071 GitHub commits and issues** from **1,805 open-source projects** that describe architectural problems encountered during software development and reference architectural solutions from SO. Each entry includes the textual description of the architectural problem used in this study, while the corresponding solution descriptions are obtained from SO.

- **`2,534_ARPs.xlsx`** – Contains **2,534 randomly selected architectural solutions** from the original set of **10,423 architectural solutions (ARPs)** curated in our previous study [10]. These solutions were paired with GitHub architectural problems to construct the **irrelevant (negative) ⟨architectural problem, solution⟩ pairs** included in the benchmark dataset.

The `results/` directory contains the following dataset:

- **`CPArchPSLinker_Results.xlsx`** – Contains the linking results generated by **CPArchPSLinker** on the **CrossPArchPSBench** benchmark dataset. Each record represents linked architectural problem-solution(s) and includes the following fields: `commit_id` (GitHub commit or issue identifier), `so_id` (Stack Overflow post identifier), `p_text` (architectural problem description), `s_text` (architectural solution text), `stage2_score` (the relevance score predicted by the model), and `stage2_rank` (the ranking position assigned to the solution for a given architectural problem).

## Requirements

The project dependencies are listed in requirements.txt.

To install all required packages, run:

```bash
pip install -r requirements.txt
```
> **Note:** For PyTorch, follow the official installation guide to ensure proper installation for your system (CPU or GPU support).


---
## 📝 Citation

```bibtex
@article{Musenga2025ArchISMiner,
  author = {Musengamana Jean de Dieu and Wenming Cao and Xinpeng Yin},
  title = {{CPArchPSLinker: Cross-Platform Linking of Architectural Solutions from Q&A Platforms to Architectural Problems in Commits and Issues}},
  journal={arXiv preprint arXiv:xxx},
  year={2026}
}
```
