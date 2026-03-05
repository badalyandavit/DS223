# Bass Model Diffusion Analysis of Generative AI Platforms

**Course:** DS223 Marketing Analytics  
**Assignment:** Homework 1 – Bass Model  
**Author:** Davit Badalyan  

This project estimates **Bass diffusion models** for generative AI platforms using web traffic adoption data.  
The analysis models the diffusion dynamics of **ChatGPT** and **DeepSeek** and compares their adoption patterns using estimated Bass parameters.

The repository contains the full **data pipeline, analysis notebook, generated outputs, and LaTeX report**.

---

# Project Structure

```
Bass_Model/
├─ requirements.txt
├─ analysis.ipynb
│
├─ data/
│  └─ similarweb_genai_platform_visits_2025.csv
│
├─ outputs/
│  ├─ bass2025_chatgpt_com_fit.png
│  ├─ bass2025_chatgpt_com_cumulative.png
│  ├─ bass2025_deepseek_com_fit.png
│  ├─ bass2025_deepseek_com_cumulative.png
│  ├─ bass2025_deepseek_com_transfer_from_chatgpt_com_fit.png
│  ├─ bass2025_chatgpt_com.csv
│  ├─ bass2025_deepseek_com.csv
│  ├─ bass2025_deepseek_com_with_transfer_from_chatgpt_com.csv
│  ├─ bass2025_chatgpt_com_preview.csv
│  ├─ bass2025_deepseek_com_preview.csv
│  └─ bass2025_parameters_scaled_millions.csv
│
├─ tables/
│  ├─ chatgpt_own_longtable.tex
│  ├─ deepseek_own_longtable.tex
│  └─ deepseek_transfer_from_chatgpt_longtable.tex
│
├─ report/
│  └─ DS223___Homework_1___Bass_Model__Davit_Badalyan/
│     ├─ main.tex
│     ├─ references.bib
│     ├─ img/
│     └─ tables/
│
└─ DS223___Homework_1___Bass_Model__Davit_Badalyan.pdf
```

---

# Directory Description

### `data/`

Contains the raw dataset used in the analysis.

- **similarweb_genai_platform_visits_2025.csv**  
  Monthly web traffic estimates for major generative AI platforms obtained from SimilarWeb.  
  These values are used as a proxy for adoption levels in the Bass diffusion model.

---

### `outputs/`

Contains all generated outputs from the analysis notebook.

These include:

**Model outputs**
- Estimated Bass model time series
- Parameter estimates
- Preview tables used for validation

**Visualizations**
- Bass model fit plots
- Cumulative adoption curves
- Cross-platform parameter transfer experiment

These outputs are later reused in the LaTeX report.

---

### `tables/`

Contains LaTeX tables automatically generated from the notebook.

- `chatgpt_own_longtable.tex`  
  Observed vs fitted Bass model values for ChatGPT.

- `deepseek_own_longtable.tex`  
  Observed vs fitted Bass model values for DeepSeek.

- `deepseek_transfer_from_chatgpt_longtable.tex`  
  DeepSeek diffusion simulation using parameters estimated from ChatGPT.

---

### `report/`

Contains the full **LaTeX source code** for the final report.

- `main.tex` – Main LaTeX document  
- `references.bib` – Bibliography file  
- `img/` – Figures used in the report  
- `tables/` – LaTeX tables imported into the report

---

### Root directory

- **analysis.ipynb**  
  Jupyter notebook performing the Bass model estimation, visualization generation, and table export.

- **requirements.txt**  
  Python dependencies required to reproduce the analysis.

- **DS223___Homework_1___Bass_Model__Davit_Badalyan.pdf**  
  Final compiled report submitted for the assignment.

---

# Setup

Install the required dependencies:

```
pip install -r requirements.txt
```

---

# Running the Analysis

Open the Jupyter notebook:

```
analysis.ipynb
```

The notebook performs:

1. Data loading and preprocessing  
2. Bass model parameter estimation  
3. Diffusion forecasting  
4. Visualization generation  
5. Export of LaTeX tables used in the report

---

# Data Source

Web traffic adoption proxy obtained from:

**SimilarWeb – GenAI Market Winners**

https://www.similarweb.com/blog/insights/marketing-insights/gen-ai-market-winners/

---

# License

This repository is provided for **academic coursework purposes**.