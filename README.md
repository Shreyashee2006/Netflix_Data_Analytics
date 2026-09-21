# Netflix Data Analytics with AI

---

## Student Information

| Field | Details |
|---|---|
| **Student Name** | Shreyashee Ghosh |
| **Program** | IBM SkillsBuild Data Analytics with AI Academic Internship |
| **Project Title** | Netflix Data Analytics with AI |

---

## Project Description

This project is a complete beginner-friendly **Data Analytics + AI** study built on the **Netflix Movies and TV Shows** dataset as part of the **IBM SkillsBuild Data Analytics with AI Academic Internship Program**.

The project covers the full data analytics lifecycle — from loading and cleaning raw data, through exploratory data analysis and visualization, to building and evaluating a Machine Learning classification model. All statistics, findings, and model results are derived directly from the actual `netflix_titles.csv` dataset.

---

## Objectives

1. Load and explore the Netflix Movies and TV Shows dataset using pandas.
2. Clean the data by handling missing values and correcting data-entry errors.
3. Perform Exploratory Data Analysis (EDA) to discover patterns and trends.
4. Create clear, well-labelled visualizations using Matplotlib and Seaborn.
5. Build a simple Machine Learning model using Scikit-learn to classify a Netflix title as a **Movie** or **TV Show**.
6. Evaluate the model using standard classification metrics.
7. Summarize key findings and propose future directions.

---

## Dataset Description

| Property | Value |
|---|---|
| **File name** | `netflix_titles.csv` |
| **Total records** | 8,807 titles |
| **Total columns** | 12 |
| **Content types** | Movies (6,131) and TV Shows (2,676) |
| **Release year range** | 1925 – 2021 |

### Dataset Source

The dataset is the publicly available **Netflix Movies and TV Shows** dataset, widely used for data analytics practice and available on [Kaggle — Netflix Movies and TV Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows).

### Dataset Columns

| Column | Description |
|---|---|
| `show_id` | Unique identifier for each title |
| `type` | Movie or TV Show |
| `title` | Name of the title |
| `director` | Director(s) |
| `cast` | Lead cast members |
| `country` | Country of production |
| `date_added` | Date added to Netflix |
| `release_year` | Original release year |
| `rating` | Content rating (e.g. TV-MA, PG-13) |
| `duration` | Duration in minutes (Movies) or seasons (TV Shows) |
| `listed_in` | Genre categories |
| `description` | Short description |

---

## Technologies Used

| Tool | Purpose |
|---|---|
| Python 3.x | Core programming language |
| Jupyter Notebook | Interactive development environment |
| pandas | Data loading, cleaning, and manipulation |
| NumPy | Numerical operations |
| Matplotlib | Base visualizations (charts, histograms, box plots) |
| Seaborn | Enhanced statistical visualizations (heatmaps, count plots) |
| Scikit-learn | Machine Learning model (Decision Tree Classifier) |

---

## Libraries Used

```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## Project Folder Structure

```
ShreyasheeGhosh_Netflix_Data_Analytics/
│
├── netflix_titles.csv                              ← Dataset (do not delete)
├── ShreyasheeGhosh_Netflix_Data_Analytics_Project.ipynb  ← Main Jupyter Notebook
├── ShreyasheeGhoshProjectReport.docx              ← Academic Project Report
├── requirements.txt                                ← Python dependencies
└── README.md                                       ← This file
```

---

## Installation Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Jupyter Notebook or JupyterLab

### Steps

1. Clone or download the project folder to your local machine.

2. Open a terminal (Command Prompt / PowerShell on Windows, or Terminal on Mac/Linux) and navigate to the project folder:

   ```bash
   cd path/to/ShreyasheeGhosh_Netflix_Data_Analytics
   ```

3. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

4. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

5. Open `ShreyasheeGhosh_Netflix_Data_Analytics_Project.ipynb` in the browser.

---

## How to Run the Notebook

1. Make sure `netflix_titles.csv` is present in the **same folder** as the notebook.
2. Open the notebook in Jupyter.
3. Run cells from top to bottom using **Shift + Enter** or use **Kernel → Restart & Run All**.
4. All charts will be displayed inline and saved as PNG files in the project folder.

> **Important:** Do not move or rename `netflix_titles.csv`. The notebook reads it from the current directory.

---

## Data Analysis Performed

| Analysis | Description |
|---|---|
| Missing Value Analysis | Identified and handled missing values in director, cast, country, rating, date_added, duration |
| Content Type Distribution | 6,131 Movies (69.7%) vs 2,676 TV Shows (30.3%) |
| Content by Release Year | Growth trend from 2000 to 2021 |
| Titles Added Over Time | Year-by-year addition count (2015–2021) |
| Top Countries | United States leads with ~2,818 titles; India #2 with ~972 |
| Genre Analysis | International Movies and Dramas are the most common genres |
| Content Ratings | TV-MA is the most common rating (3,207 titles) |
| Movie Duration | Average duration ~99.6 minutes; range 3–312 minutes |
| TV Show Seasons | ~67% of TV Shows have only 1 season |
| Seasonal Heatmap | July, October, and December show peak content additions |

---

## Machine Learning Component

| Property | Value |
|---|---|
| **Task** | Binary Classification (Movie vs TV Show) |
| **Algorithm** | Decision Tree Classifier (max_depth=5) |
| **Features used** | `release_year`, `rating` (encoded), `year_added` |
| **Target variable** | `type` (Movie = 0, TV Show = 1) |
| **Train/Test split** | 80% training (7,035 samples) / 20% testing (1,759 samples) |
| **Accuracy** | 72.88% |
| **Weighted Precision** | 0.7077 |
| **Weighted Recall** | 0.7288 |
| **Weighted F1-Score** | 0.7052 |

> **Note on fairness:** The `duration` column was deliberately excluded from features because it directly encodes the content type (movies show "min", TV shows show "Seasons"). Including it would create a trivially perfect but meaningless model.

---

## Key Findings

1. Netflix's library is dominated by **Movies (~70%)**, though TV Show production has grown steadily since 2017.
2. The **United States** produces the most Netflix content (~32% of all titles).
3. **TV-MA** (Mature Audience) is the most common rating — over 60% of content targets adults.
4. The most popular genre tags are **International Movies**, **Dramas**, and **Comedies**.
5. Netflix's content additions **peaked in 2019** with over 2,000 titles added that year.
6. Average movie duration is approximately **99.6 minutes** (standard Hollywood runtime).
7. ~**67% of TV Shows** have only **1 season**, reflecting Netflix's preference for limited-run international series.
8. Content additions show **seasonal peaks in July, October, and December**.

---

## Conclusion

This project successfully demonstrated a complete data analytics and AI pipeline on a real-world dataset. Using the Netflix Movies and TV Shows dataset of 8,807 titles, we performed thorough data cleaning, exploratory analysis, visualization, and machine learning. The Decision Tree Classifier achieved **72.88% accuracy** in predicting whether a title is a Movie or TV Show without using any directly revealing features.

---

## Future Scope

- **Sentiment Analysis** on title descriptions using NLP
- **Multi-class Genre Prediction** using Random Forest or Gradient Boosting
- **Content Recommendation System** based on genre, country, and rating
- **Interactive Dashboard** using Streamlit or Dash
- **Time Series Forecasting** of future content additions
- **Deep Learning** text classification using BERT
- **Network Analysis** of cast and director co-appearances
- **Comparison with competitors** (Amazon Prime, Disney+)

---

*Project completed by **Shreyashee Ghosh** — IBM SkillsBuild Data Analytics with AI Academic Internship*
