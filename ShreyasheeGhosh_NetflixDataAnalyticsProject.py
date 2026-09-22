# =============================================================================
# Netflix Data Analytics with AI
# =============================================================================
# Student Name  : Shreyashee Ghosh
# Program       : IBM SkillsBuild Data Analytics with AI Academic Internship
# Project Title : Netflix Data Analytics with AI
# Dataset       : Netflix Movies and TV Shows (netflix_titles.csv)
# Tools Used    : Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
# =============================================================================
#
# Project Objective:
#   - Load and understand the structure of the Netflix dataset.
#   - Clean the data by handling missing values and inconsistencies.
#   - Perform Exploratory Data Analysis (EDA) to uncover patterns and trends.
#   - Create clear, well-labelled visualizations using Matplotlib and Seaborn.
#   - Build a simple Machine Learning model (Decision Tree Classifier) to
#     predict whether a Netflix title is a Movie or a TV Show.
#   - Evaluate the model using standard classification metrics.
#   - Summarize key findings and suggest future directions.
#
# Dataset Description:
#   The dataset has 8,807 rows and 12 columns:
#   show_id      - Unique identifier for each title
#   type         - Movie or TV Show
#   title        - Name of the title
#   director     - Director(s) of the title
#   cast         - Lead cast members
#   country      - Country of production
#   date_added   - Date added to Netflix
#   release_year - Original release year
#   rating       - Content rating (e.g. TV-MA, PG-13)
#   duration     - Duration in minutes (Movies) or number of seasons (TV Shows)
#   listed_in    - Genre categories
#   description  - Short description of the title
# =============================================================================


# =============================================================================
# SECTION 5 — Importing Libraries
# =============================================================================
# We import all the Python libraries needed for this project.
#   pandas     : loading and manipulating data
#   numpy      : numerical operations
#   matplotlib : creating basic visualizations
#   seaborn    : enhanced statistical visualizations
#   scikit-learn: building and evaluating the Machine Learning model

# Standard data manipulation libraries
import pandas as pd
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Machine Learning libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# Display settings
pd.set_option('display.max_columns', 15)
pd.set_option('display.max_colwidth', 50)

# Set a consistent plot style
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['font.size'] = 12

print("All libraries imported successfully!")


# =============================================================================
# SECTION 6 — Loading the Dataset
# =============================================================================
# We load the Netflix dataset from the CSV file into a pandas DataFrame.
# A DataFrame is like a table with rows and columns — similar to a spreadsheet.

# Load the dataset
df = pd.read_csv('netflix_titles.csv')

print(f"Dataset loaded successfully!")
print(f"Number of rows    : {df.shape[0]}")
print(f"Number of columns : {df.shape[1]}")

# Display the first 5 rows of the dataset
print("\nFirst 5 rows of the dataset:")
print(df.head())


# =============================================================================
# SECTION 7 — Understanding the Dataset
# =============================================================================
# Before cleaning and analyzing the data, it is important to understand its
# structure: columns, data types, missing values, and duplicate records.

# Shape of the dataset
print(f"\nDataset shape: {df.shape[0]} rows x {df.shape[1]} columns")

# Column names
print("\nColumn names:")
print(list(df.columns))

# Data types of each column
print("\nData types:")
print(df.dtypes)

# Statistical summary for numerical columns
print("\nStatistical summary:")
print(df.describe())

# Missing values analysis
print("\nMissing values per column:")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
print(missing_df[missing_df['Missing Count'] > 0])

# Check for duplicate rows
duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")

# --- Observations ---
# - The dataset has 8,807 titles and 12 columns.
# - 'director' has the most missing values: 2,634 (29.9%) — many TV shows
#   do not have a single credited director.
# - 'cast' is missing for 825 rows (9.4%) and 'country' for 831 rows (9.4%).
# - 'date_added' has only 10 missing values, 'rating' has 4, 'duration' has 3.
# - There are NO duplicate rows in the dataset.


# =============================================================================
# SECTION 8 — Data Cleaning and Preprocessing
# =============================================================================
# Data cleaning steps:
#  1. Strip extra whitespace from string columns.
#  2. Fix 3 rows where 'rating' accidentally stores duration values.
#  3. Drop the 3 rows where 'duration' is missing (same erroneous rows).
#  4. Fill missing values in director, cast, country, rating with 'Unknown'.
#  5. Parse date_added to datetime; extract year_added and month_added.
#  6. Extract numeric duration (minutes for Movies, seasons for TV Shows).

# ── Step 1: Strip whitespace from object columns ──────────────────────────
str_cols = df.select_dtypes(include='object').columns
df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())

print("Whitespace stripped from all string columns.")

# ── Step 2: Fix data-entry errors in 'rating' column ─────────────────────
# Three rows have duration values ('74 min', '84 min', '66 min') stored in
# the rating column — they are data-entry mistakes.
bad_ratings = ['66 min', '74 min', '84 min']
print("\nRows with bad rating values:")
print(df[df['rating'].isin(bad_ratings)][['title', 'rating', 'duration']])

# Set these erroneous rating values to NaN
df.loc[df['rating'].isin(bad_ratings), 'rating'] = np.nan
print("\nBad rating values replaced with NaN.")

# ── Step 3: Drop rows where 'duration' is missing (only 3 rows) ──────────
df.dropna(subset=['duration'], inplace=True)
print(f"\nRows after dropping missing duration: {len(df)}")

# ── Step 4: Fill missing values in text/categorical columns ───────────────
# We use 'Unknown' instead of dropping rows so we keep all 8,804 records.
df['director'] = df['director'].fillna('Unknown')
df['cast']     = df['cast'].fillna('Unknown')
df['country']  = df['country'].fillna('Unknown')
df['rating']   = df['rating'].fillna('Unknown')

print("\nMissing values after filling:")
print(df.isnull().sum())

# ── Step 5: Parse date_added to datetime; extract year and month ──────────
df['date_added'] = pd.to_datetime(df['date_added'], format='%B %d, %Y', errors='coerce')
df['year_added']  = df['date_added'].dt.year
df['month_added'] = df['date_added'].dt.month

print("\nSample of parsed date_added:")
print(df[['date_added', 'year_added', 'month_added']].head())

# ── Step 6: Extract numeric duration ──────────────────────────────────────
# For Movies   -> '90 min'    -> 90
# For TV Shows -> '2 Seasons' -> 2
df['duration_numeric'] = df['duration'].str.extract(r'(\d+)').astype(float)

print("\nMovies - duration_numeric sample:")
print(df[df['type'] == 'Movie'][['title', 'duration', 'duration_numeric']].head())

print("\nTV Shows - duration_numeric sample:")
print(df[df['type'] == 'TV Show'][['title', 'duration', 'duration_numeric']].head())

# ── Final check after cleaning ────────────────────────────────────────────
print(f"\nFinal dataset shape after cleaning: {df.shape}")
remaining_missing = df.isnull().sum()[df.isnull().sum() > 0]
if len(remaining_missing) > 0:
    print(f"Missing values remaining:\n{remaining_missing}")
else:
    print("No critical missing values remain.")

# --- Cleaning Summary ---
# Column    | Issue                         | Treatment
# director  | 2,634 missing                 | Filled with 'Unknown'
# cast      | 825 missing                   | Filled with 'Unknown'
# country   | 831 missing                   | Filled with 'Unknown'
# rating    | 4 missing + 3 data-entry errors | Errors to NaN, all NaN -> 'Unknown'
# duration  | 3 missing                     | Rows dropped (unusable)
# date_added| 10 missing, stored as string  | Parsed to datetime


# =============================================================================
# SECTION 9 & 10 — Exploratory Data Analysis and Data Visualizations
# =============================================================================
# EDA visually and statistically explores the data to find patterns before
# building any model. We cover 10 analyses with charts.


# ── 9.1 Distribution of Movies vs TV Shows ────────────────────────────────
print("\n--- 9.1 Distribution of Movies vs TV Shows ---")

type_counts = df['type'].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Bar chart
colors = ['#e50914', '#564d4d']
axes[0].bar(type_counts.index, type_counts.values, color=colors, edgecolor='white', linewidth=0.8)
axes[0].set_title('Number of Movies vs TV Shows on Netflix', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Content Type', fontsize=12)
axes[0].set_ylabel('Number of Titles', fontsize=12)
for i, v in enumerate(type_counts.values):
    axes[0].text(i, v + 40, str(v), ha='center', fontsize=12, fontweight='bold')

# Pie chart
axes[1].pie(
    type_counts.values,
    labels=type_counts.index,
    autopct='%1.1f%%',
    colors=colors,
    startangle=90,
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
)
axes[1].set_title('Proportion of Movies vs TV Shows', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('chart_01_type_distribution.png', dpi=120, bbox_inches='tight')
plt.show()

print(f"Movies  : {type_counts['Movie']}  ({type_counts['Movie']/len(df)*100:.1f}%)")
print(f"TV Shows: {type_counts['TV Show']} ({type_counts['TV Show']/len(df)*100:.1f}%)")
# Chart Explanation:
# The Netflix library is dominated by Movies (~69.7%). TV Shows account for
# ~30.3%. Netflix has historically added more movies than shows.


# ── 9.2 Netflix Content by Release Year ───────────────────────────────────
print("\n--- 9.2 Netflix Content by Release Year ---")

# Focus on years from 2000 onwards for a cleaner chart
release_year_data = df[df['release_year'] >= 2000].groupby(['release_year', 'type']).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(13, 5))
release_year_data.plot(kind='bar', ax=ax, color=['#e50914', '#564d4d'], edgecolor='white', linewidth=0.5)

ax.set_title('Netflix Titles by Release Year (2000–2021)', fontsize=14, fontweight='bold')
ax.set_xlabel('Release Year', fontsize=12)
ax.set_ylabel('Number of Titles', fontsize=12)
ax.legend(title='Content Type', fontsize=11)
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('chart_02_release_year.png', dpi=120, bbox_inches='tight')
plt.show()

# Chart Explanation:
# Titles grew dramatically from 2014 onwards. The peak for Movies is
# 2017-2019. The drop in 2021 is because the dataset was collected mid-year.


# ── 9.3 Titles Added to Netflix Over Time ─────────────────────────────────
print("\n--- 9.3 Titles Added to Netflix Over Time ---")

titles_by_year = df.dropna(subset=['year_added']).groupby(['year_added', 'type']).size().unstack(fill_value=0)
titles_by_year = titles_by_year[titles_by_year.index >= 2015]  # Focus on meaningful range

fig, ax = plt.subplots(figsize=(12, 5))
titles_by_year['Movie'].plot(ax=ax, marker='o', color='#e50914', linewidth=2.5, label='Movie')
titles_by_year['TV Show'].plot(ax=ax, marker='s', color='#564d4d', linewidth=2.5, label='TV Show')

ax.set_title('Number of Titles Added to Netflix Per Year (2015–2021)', fontsize=14, fontweight='bold')
ax.set_xlabel('Year Added to Netflix', fontsize=12)
ax.set_ylabel('Number of Titles Added', fontsize=12)
ax.legend(title='Content Type', fontsize=11)
ax.set_xticks(titles_by_year.index)
ax.tick_params(axis='x', rotation=0)
plt.tight_layout()
plt.savefig('chart_03_titles_added_over_time.png', dpi=120, bbox_inches='tight')
plt.show()

print("Titles added per year:")
print(titles_by_year)

# Chart Explanation:
# Netflix additions peaked in 2019 (2,016 titles). A decline in 2020-2021
# is partly due to COVID-19 production disruptions and dataset cutoff date.


# ── 9.4 Top 15 Countries Producing Netflix Content ────────────────────────
print("\n--- 9.4 Top 15 Countries Producing Netflix Content ---")

# Use the first listed country for each title
df['primary_country'] = df['country'].str.split(',').str[0].str.strip()

top_countries = (
    df[df['primary_country'] != 'Unknown']['primary_country']
    .value_counts()
    .head(15)
)

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x=top_countries.values, y=top_countries.index, palette='Reds_r', ax=ax)

ax.set_title('Top 15 Countries by Number of Netflix Titles', fontsize=14, fontweight='bold')
ax.set_xlabel('Number of Titles', fontsize=12)
ax.set_ylabel('Country', fontsize=12)

for i, v in enumerate(top_countries.values):
    ax.text(v + 10, i, str(v), va='center', fontsize=10)

plt.tight_layout()
plt.savefig('chart_04_top_countries.png', dpi=120, bbox_inches='tight')
plt.show()

# Chart Explanation:
# The United States leads with ~2,800+ titles, then India (~972) and UK (~419).
# South Korea, Japan, Spain, and France highlight Netflix's global strategy.


# ── 9.5 Most Common Genres / Categories ───────────────────────────────────
print("\n--- 9.5 Most Common Genres ---")

# Expand multi-genre entries — each title can belong to multiple genres
all_genres = df['listed_in'].str.split(', ').explode().str.strip()
top_genres = all_genres.value_counts().head(15)

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x=top_genres.values, y=top_genres.index, palette='Blues_r', ax=ax)

ax.set_title('Top 15 Most Common Genres on Netflix', fontsize=14, fontweight='bold')
ax.set_xlabel('Number of Titles', fontsize=12)
ax.set_ylabel('Genre', fontsize=12)

for i, v in enumerate(top_genres.values):
    ax.text(v + 10, i, str(v), va='center', fontsize=10)

plt.tight_layout()
plt.savefig('chart_05_top_genres.png', dpi=120, bbox_inches='tight')
plt.show()

# Chart Explanation:
# International Movies is the most common tag (~2,752), then Dramas (~2,427)
# and Comedies (~1,674). This reflects Netflix's global library strategy.


# ── 9.6 Content Ratings Distribution ──────────────────────────────────────
print("\n--- 9.6 Content Ratings Distribution ---")

rating_counts = (
    df[df['rating'] != 'Unknown']['rating']
    .value_counts()
    .reset_index()
)
rating_counts.columns = ['rating', 'count']

fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(data=rating_counts, x='rating', y='count', palette='viridis', ax=ax)

ax.set_title('Distribution of Content Ratings on Netflix', fontsize=14, fontweight='bold')
ax.set_xlabel('Content Rating', fontsize=12)
ax.set_ylabel('Number of Titles', fontsize=12)
ax.tick_params(axis='x', rotation=45)

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('chart_06_ratings.png', dpi=120, bbox_inches='tight')
plt.show()

# Chart Explanation:
# TV-MA (Mature Audience) is the most common rating with 3,207 titles.
# TV-14 follows with 2,160. Netflix's primary audience is adults.


# ── 9.7 Movie Duration Distribution ───────────────────────────────────────
print("\n--- 9.7 Movie Duration Distribution ---")

movie_durations = df[df['type'] == 'Movie']['duration_numeric'].dropna()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(movie_durations, bins=40, color='#e50914', edgecolor='white', linewidth=0.5)
axes[0].axvline(movie_durations.mean(), color='black', linestyle='--', linewidth=1.5,
                label=f'Mean: {movie_durations.mean():.0f} min')
axes[0].axvline(movie_durations.median(), color='navy', linestyle='--', linewidth=1.5,
                label=f'Median: {movie_durations.median():.0f} min')
axes[0].set_title('Distribution of Movie Durations', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Duration (minutes)', fontsize=12)
axes[0].set_ylabel('Number of Movies', fontsize=12)
axes[0].legend(fontsize=11)

# Box plot
axes[1].boxplot(movie_durations, vert=True, patch_artist=True,
                boxprops=dict(facecolor='#e50914', color='black'),
                medianprops=dict(color='white', linewidth=2))
axes[1].set_title('Box Plot of Movie Durations', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Duration (minutes)', fontsize=12)
axes[1].set_xticks([])

plt.tight_layout()
plt.savefig('chart_07_movie_duration.png', dpi=120, bbox_inches='tight')
plt.show()

print(f"Movie duration statistics:")
print(f"  Mean   : {movie_durations.mean():.1f} minutes")
print(f"  Median : {movie_durations.median():.0f} minutes")
print(f"  Min    : {movie_durations.min():.0f} minutes")
print(f"  Max    : {movie_durations.max():.0f} minutes")
print(f"  Std Dev: {movie_durations.std():.1f} minutes")

# Chart Explanation:
# Duration is roughly bell-shaped, centred around 98-100 minutes.
# Most movies are 75-125 minutes. The average is ~99.6 minutes.


# ── 9.8 TV Show Seasons Distribution ──────────────────────────────────────
print("\n--- 9.8 TV Show Seasons Distribution ---")

show_seasons = df[df['type'] == 'TV Show']['duration_numeric'].dropna()
season_counts = show_seasons.value_counts().sort_index().head(10)

fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(season_counts.index.astype(int), season_counts.values, color='#564d4d', edgecolor='white', linewidth=0.8)

ax.set_title('TV Show Distribution by Number of Seasons', fontsize=14, fontweight='bold')
ax.set_xlabel('Number of Seasons', fontsize=12)
ax.set_ylabel('Number of TV Shows', fontsize=12)
ax.set_xticks(range(1, 11))

for i, v in zip(season_counts.index.astype(int), season_counts.values):
    ax.text(i, v + 15, str(v), ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('chart_08_tv_seasons.png', dpi=120, bbox_inches='tight')
plt.show()

print(f"TV Show season statistics:")
print(f"  Total TV Shows : {len(show_seasons)}")
print(f"  1-season shows : {int(season_counts[1])} ({season_counts[1]/len(show_seasons)*100:.1f}%)")
print(f"  Mean seasons   : {show_seasons.mean():.2f}")

# Chart Explanation:
# About 67% of TV Shows on Netflix have only 1 season. Netflix favours
# newer or limited-run international series.


# ── 9.9 Heatmap: Content Added by Month and Year ─────────────────────────
print("\n--- 9.9 Heatmap: Content Added by Month and Year ---")

month_year = (
    df.dropna(subset=['year_added', 'month_added'])
    .groupby(['year_added', 'month_added'])
    .size()
    .unstack(fill_value=0)
)
month_year = month_year[month_year.index >= 2016]
month_year.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

fig, ax = plt.subplots(figsize=(13, 5))
sns.heatmap(
    month_year,
    annot=True,
    fmt='d',
    cmap='YlOrRd',
    linewidths=0.5,
    ax=ax,
    cbar_kws={'label': 'Number of Titles Added'}
)
ax.set_title('Number of Titles Added to Netflix by Month and Year', fontsize=14, fontweight='bold')
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Year', fontsize=12)

plt.tight_layout()
plt.savefig('chart_09_heatmap_month_year.png', dpi=120, bbox_inches='tight')
plt.show()

# Chart Explanation:
# July, October, and December tend to have higher additions, timed around
# summer breaks, Halloween, and the holiday season. 2018-2019 are the darkest
# cells confirming peak content-addition years.


# ── 9.10 Content Ratings: Movies vs TV Shows (Side-by-Side) ──────────────
print("\n--- 9.10 Content Ratings: Movies vs TV Shows ---")

# Filter to well-known ratings
known_ratings = ['TV-MA', 'TV-14', 'TV-PG', 'R', 'PG-13', 'TV-Y7', 'TV-Y', 'PG', 'TV-G', 'NR', 'G']
rating_type_df = df[df['rating'].isin(known_ratings)]

fig, ax = plt.subplots(figsize=(13, 5))
sns.countplot(
    data=rating_type_df,
    x='rating',
    hue='type',
    order=known_ratings,
    palette=['#e50914', '#564d4d'],
    ax=ax
)

ax.set_title('Content Ratings: Movies vs TV Shows', fontsize=14, fontweight='bold')
ax.set_xlabel('Content Rating', fontsize=12)
ax.set_ylabel('Number of Titles', fontsize=12)
ax.legend(title='Content Type', fontsize=11)
ax.tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('chart_10_ratings_by_type.png', dpi=120, bbox_inches='tight')
plt.show()

# Chart Explanation:
# Both types are concentrated in TV-MA and TV-14. Movies also have R and
# PG-13 (cinema-specific), while TV Shows spread more evenly across TV
# ratings (TV-Y to TV-MA).


# =============================================================================
# SECTION 11 — Key Data Analysis Findings
# =============================================================================
# Based on the EDA above, the most important findings are:
#
#  1. 8,807 titles total: 6,131 Movies (69.7%) and 2,676 TV Shows (30.3%).
#  2. United States leads with ~2,818 titles; India ~972; UK ~419.
#  3. TV-MA is the most common rating (3,207 titles) — Netflix targets adults.
#  4. Top genres: International Movies (~2,752), Dramas (~2,427), Comedies (~1,674).
#  5. Content additions peaked in 2019 with over 2,000 titles that year.
#  6. Average movie duration ~99.6 minutes; range 3–312 minutes.
#  7. ~67% of TV Shows have only 1 season — Netflix favours limited-run series.
#  8. Content additions peak in July, October, and December (seasonal pattern).
#  9. Release years span 1925–2021; majority released after 2015.
# 10. 'director' has 29.9% missingness — mainly because TV shows lack one.

print("\n--- Section 11: Key Findings printed above in comments ---")


# =============================================================================
# SECTION 12 — Machine Learning / AI Component
# =============================================================================
# Task   : Binary Classification — predict Movie vs TV Show
# Model  : Decision Tree Classifier (scikit-learn, max_depth=5)
# Features: release_year, rating (label-encoded), year_added
# Target : type (Movie=0, TV Show=1)
#
# IMPORTANT — No Data Leakage:
#   The 'duration' column is deliberately excluded because it directly encodes
#   the answer ('min' for Movies, 'Seasons' for TV Shows). Using it would
#   give perfect but meaningless results.

# ── Step 1: Prepare the ML dataset ────────────────────────────────────────

# Select features and target
ml_df = df[['type', 'release_year', 'rating', 'year_added']].copy()

# Drop rows with missing year_added (only a few rows)
ml_df.dropna(subset=['year_added'], inplace=True)

# Convert year_added to integer
ml_df['year_added'] = ml_df['year_added'].astype(int)

# Encode the 'rating' column: convert text ratings to numbers
le_rating = LabelEncoder()
ml_df['rating_encoded'] = le_rating.fit_transform(ml_df['rating'])

# Encode the target: Movie = 0, TV Show = 1
le_type = LabelEncoder()
ml_df['type_encoded'] = le_type.fit_transform(ml_df['type'])

print("\nLabel encoding mapping for 'type':")
for label, code in zip(le_type.classes_, range(len(le_type.classes_))):
    print(f"  {label} -> {code}")

print(f"\nML dataset shape: {ml_df.shape}")
print(ml_df[['type', 'type_encoded', 'release_year', 'rating', 'rating_encoded', 'year_added']].head())

# ── Step 2: Define features (X) and target (y) ────────────────────────────
X = ml_df[['release_year', 'rating_encoded', 'year_added']]
y = ml_df['type_encoded']

print(f"\nFeatures (X) shape : {X.shape}")
print(f"Target (y) shape   : {y.shape}")
print(f"\nClass distribution:")
print(y.value_counts().rename(index={0: 'Movie', 1: 'TV Show'}))

# ── Step 3: Split into Training and Testing sets ───────────────────────────
# 80% for training, 20% for testing; random_state=42 ensures reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\nTraining set size : {X_train.shape[0]} samples")
print(f"Testing set size  : {X_test.shape[0]} samples")

# ── Step 4: Train the Decision Tree Classifier ────────────────────────────
# max_depth=5 limits tree complexity and helps prevent overfitting
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

print("\nDecision Tree model trained successfully!")

# ── Step 5: Make Predictions ──────────────────────────────────────────────
y_pred = dt_model.predict(X_test)

print("\nPredictions generated on the test set.")
print(f"First 10 actual    : {list(y_test[:10])}")
print(f"First 10 predicted : {list(y_pred[:10])}")


# =============================================================================
# SECTION 13 — Model Evaluation
# =============================================================================
# Metrics explained:
#   Accuracy  — What % of all predictions are correct?
#   Precision — Of titles predicted as TV Show, how many actually are?
#   Recall    — Of all actual TV Shows, how many did the model find?
#   F1-Score  — Harmonic mean of Precision and Recall.
#   Confusion Matrix — Shows correct vs incorrect per class.

# ── Compute evaluation metrics ─────────────────────────────────────────────
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall    = recall_score(y_test, y_pred, average='weighted')
f1        = f1_score(y_test, y_pred, average='weighted')

print("=" * 42)
print("         MODEL EVALUATION RESULTS")
print("=" * 42)
print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1-Score  : {f1:.4f}")
print("=" * 42)

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=le_type.classes_))

# ── Confusion Matrix Visualization ─────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Reds',
    xticklabels=le_type.classes_,
    yticklabels=le_type.classes_,
    linewidths=0.5,
    ax=ax
)

ax.set_title('Confusion Matrix — Decision Tree Classifier', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('Actual Label', fontsize=12)

plt.tight_layout()
plt.savefig('chart_11_confusion_matrix.png', dpi=120, bbox_inches='tight')
plt.show()

print("\nHow to read the confusion matrix:")
print(f"  Correctly predicted Movies   : {cm[0][0]}")
print(f"  Movies predicted as TV Show  : {cm[0][1]}")
print(f"  TV Shows predicted as Movie  : {cm[1][0]}")
print(f"  Correctly predicted TV Shows : {cm[1][1]}")

# ── Feature Importances ─────────────────────────────────────────────────────
feature_names = ['release_year', 'rating_encoded', 'year_added']
importances = dt_model.feature_importances_

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(feature_names, importances, color=['#e50914', '#564d4d', '#999999'])

ax.set_title('Feature Importances — Decision Tree', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance Score', fontsize=12)
ax.set_ylabel('Feature', fontsize=12)

for bar, imp in zip(bars, importances):
    ax.text(imp + 0.005, bar.get_y() + bar.get_height() / 2,
            f'{imp:.4f}', va='center', fontsize=11)

plt.tight_layout()
plt.savefig('chart_12_feature_importance.png', dpi=120, bbox_inches='tight')
plt.show()

print("\nFeature importances (sorted by importance):")
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    print(f"  {name:<20}: {imp:.4f}")

# Model Evaluation Notes (Beginner-Friendly):
#   - The model achieves moderate accuracy. This is expected because we
#     deliberately excluded 'duration' (which would trivially solve the task).
#   - 'rating_encoded' is the most important feature (~73%) because Movies use
#     cinema ratings (R, PG-13) while TV Shows use TV ratings (TV-MA, TV-14).
#   - Most errors occur in predicting TV Shows — Movies outnumber TV Shows
#     ~2.3:1 (class imbalance).
#   - This is an educational model, not intended for production use.


# =============================================================================
# SECTION 14 — Important Insights
# =============================================================================
# Content Mix:
#   Netflix has ~2.3x more Movies than TV Shows. TV Show production has been
#   growing steadily, especially after 2017.
#
# Global Reach:
#   The US dominates with ~32% of all titles. India, UK, Japan, and South
#   Korea contribute significantly, confirming Netflix's global strategy.
#
# Genre Trends:
#   International Movies and Dramas are the most common genres. The high
#   'International' tag count shows Netflix curates for a global audience.
#
# Growth Pattern:
#   The library grew explosively 2016-2019. The slowdown post-2019 coincides
#   with global pandemic effects on production.
#
# Duration Patterns:
#   Average movie ~100 minutes (standard Hollywood runtime). 67% of TV Shows
#   have exactly 1 season — Netflix favours international limited series.
#
# Audience Targeting:
#   Over 60% of content is rated TV-MA or TV-14, showing strong focus on
#   mature adult content.
#
# Machine Learning:
#   A Decision Tree can distinguish Movies from TV Shows using only
#   release_year, rating, and year_added — showing Movies and TV Shows have
#   distinct statistical profiles.

print("\n--- Section 14: Insights summarized above in comments ---")


# =============================================================================
# SECTION 15 — Conclusion
# =============================================================================
# In this project we performed a complete Data Analytics + AI study:
#   1. Loaded and explored 8,807 Netflix titles.
#   2. Cleaned data: missing values, data-entry errors, feature engineering.
#   3. EDA across 10 dimensions with 12 visualizations.
#   4. Identified 10 key patterns and trends.
#   5. Built a Decision Tree Classifier (Movie vs TV Show).
#   6. Evaluated with Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
#   7. Summarized findings in beginner-friendly language.
#
# Key Takeaway:
#   Netflix's library is large, diverse, and globally-oriented — dominated by
#   Movies for adult audiences, with growing emphasis on international limited
#   TV series. Even a simple ML model can learn meaningful patterns from
#   structured data.

print("\n--- Section 15: Conclusion printed above in comments ---")


# =============================================================================
# SECTION 16 — Future Scope
# =============================================================================
#  1. Sentiment Analysis on Descriptions — NLP on description column.
#  2. Multi-class Genre Prediction — Random Forest or Gradient Boosting.
#  3. Content Recommendation System — content-based recommender.
#  4. Interactive Dashboard — Streamlit or Dash.
#  5. Time Series Forecasting — predict future title additions.
#  6. Handle Class Imbalance — apply SMOTE to improve TV Show recall.
#  7. Deep Learning — BERT-based description classification.
#  8. Competitive Analysis — compare with Amazon Prime / Disney+ datasets.

print("\n--- Section 16: Future Scope printed above in comments ---")
print("\n" + "=" * 60)
print("  Project completed by: Shreyashee Ghosh")
print("  IBM SkillsBuild Data Analytics with AI Academic Internship")
print("=" * 60)
