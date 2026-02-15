# Artificial-Intelligence

## Quantitative Risk & Machine Learning Portfolio

This repository serves as a professional portfolio demonstrating the technical proficiency, engineering standards, and strategic methodology developed during my Level 7 Artificial Intelligence Apprenticeship.

> **Confidentiality & Compliance Notice:**  
> This portfolio does not contain, reference, or replicate any proprietary methodologies, confidential data, or internal processes from current or former employers. All logic presented here is based on publicly available regulatory guidelines (e.g., Basel III/IV, PRA SS1/23) and standard academic machine learning practices.

---

#### Assessment Method 1: Project Report with Presentation

**Grade Achieved:** Distinction

A comprehensive work-based project documented in a technical report of approximately 5,000 words (+/- 10%), followed by a formal presentation and supplementary questioning.

**Report Structure:**
Written Report was on highly sensitive data and methodology, a summary framework is provided below: 
- **Introduction & Background:** Outline of the business problem (unlocking capital capacity and strengthening balance sheet strategy).
- **Methods & Justification:** Survey of potential alternatives and justification for chosen ML architectures (Logistic Regression vs. XGBoost vs. Ensemble).
- **Scope & KPIs:** Defining predictive power targets (gAUC/ROC) and regulatory compliance benchmarks.
- **Data Selection & Pre-processing:** Integration of multi-source loan characteristics and borrower behaviour data.
- **Implementation:** Performance metrics and transition from SAS to Python-based automated optimisation.
- **Results:** Quantitative analysis of RWA efficiency and default detection improvements.
- **Discussion & Conclusions:** Summary of findings, implications for balance sheet strategy, and caveats/limitations.
- **Statistical Rigour:** Inclusion of uncertainty, bias, and error estimates as appropriate for PRA-regulated models.
- **KSB Mapping:** Direct mapping of project activities to the Knowledge, Skills, and Behaviours of a Senior Data Scientist.

**Presentation (75 Minutes):**
- 30-minute presentation covering the high-level summary, practical application of knowledge, and business recommendations.
- 45-minute technical Q&A assessing the depth of understanding and ability to justify technical choices.
---

#### Assessment Method 2: Professional Discussion

**Grade Achieved:** Distinction

A 90-minute structured discussion underpinned by a portfolio of evidence.

- **Leadership:** Mentoring junior analysts and leading technical migrations.
- **Stakeholder Management:** Translating complex outputs for Finance, Audit, and Legal teams.
- **Continuous Improvement:** Evidence of independent study in MLOps and Enterprise Python.
- **Knowledge Transfer:** Established Python adoption frameworks (Teams channels, shared repositories, presentation materials) accelerating technical capability across a 60-person Quant Analytics function.
- **Quality Assurance:** Conducted peer reviews for 4 internal colleagues and 5 external contractors, ensuring code quality and adherence to enterprise Python standards.
- **Cross-Functional Collaboration:** Navigated complex stakeholder requirements across IVT, Finance, Audit, and Legal teams to balance technical innovation with regulatory constraints.

---

#### Assessment Method 3: Technical Test (Knowledge Test)

**Grade Achieved:** Distinction

A rigorous, closed-book examination designed to test technical knowledge and skills that cannot be fully assessed through project work alone. 

**Test Format & Administration:**
- **Duration:** 100 minutes.
- **Format:** 4 long-response questions based on complex technical scenarios.
- **Conditions:** Closed-book, invigilated exam environment (no reference materials allowed).

**Core Competencies Tested:**
- Deep theoretical understanding of AI/ML algorithms and their appropriate application.
- Problem-solving in unseen technical scenarios.
- Evaluation of technical trade-offs and architectural decisions without access to documentation.

---

### 📊 Technical Implementation: Risk Benchmarking Framework

This repository provides a clean-room implementation of the technical workflow used during the apprenticeship project.

**Portfolio Scope:** Applied to a £10bn+ mortgage portfolio (2010-2015 vintage), informing both PPGD and LGD regulatory frameworks.

---

#### ⚙️ Phase 1: Data Preparation

- **Data Integration:** Combining disparate sources to create a unified view of risk.
- **EDA & Cleansing:** Automated outlier detection and missing value imputation.
- **Partitioning:** Stratified training/testing splits to handle imbalanced default classes.

---

#### 🛠 Phase 2: Model Development

- **Variable Selection:** Utilising Information Value (IV) and Correlation matrices for parsimonious feature sets.
- **Benchmarking:** Implementation of Logistic Regression (Baseline), XGBoost (Predictive Power), and Ensemble Methods (Stability).
- **Hyperparameter Optimisation:** Implemented automated grid search and cross-validation frameworks, reducing manual calibration effort by 70% whilst improving model stability and reproducibility.

---

#### 🧪 Phase 3: Validation & Soundness

- **Qualitative:** Assessing conceptual soundness and statistical assumptions.
- **Quantitative:** Measuring Predictive Power (Gini/ROC), Explanatory Adequacy, and Population Stability (PSI).
- **Ongoing Performance Monitoring:** Established quarterly model monitoring framework for 10+ mortgage risk models, delivering RAG-status reports to senior leadership and identifying model drift requiring threshold adjustments.

---

### 📚 References

- **AI Apprenticeship Standard:** [ST0763](https://skillsengland.education.gov.uk/media/3685/st0763_artificial-intelligence-ai-data-specialist_l7_ap-for-publication_qm.pdf).
- **Regulatory Guidelines:** [PRA SS1/23 - Model Risk Management](https://www.bankofengland.co.uk/-/media/boe/files/prudential-regulation/supervisory-statement/2023/ss123.pdf).
- **Mathematical Model Framework:** [Basel III/IV Capital Requirements](https://www.bis.org/bcbs/basel3.htm).

> **Note:** This repository is intended for demonstration purposes only.
