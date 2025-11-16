# 🦠 COVID-19 Clinical Trials Exploratory Data Analysis (EDA)

A comprehensive Exploratory Data Analysis project on COVID-19 Clinical Trials dataset using Python and Pandas. This project performs in-depth analysis of clinical trials data, including data cleaning, visualization, and statistical insights.

## 📊 Project Overview

This project analyzes COVID-19 clinical trials data from ClinicalTrials.gov, providing insights into:
- Trial statuses and phases distribution
- Geographic distribution of trials
- Study types and designs
- Demographic analysis (age groups, gender)
- Funding sources
- Temporal trends
- Relationships between different variables

## 🎯 Features

### Data Processing & Cleaning
- ✅ Missing data analysis and handling
- ✅ Data type conversion and standardization
- ✅ Duplicate removal
- ✅ Country extraction from locations
- ✅ Age group categorization

### Analysis & Visualizations
- 📈 **Status Distribution** - Overview of trial statuses (Recruiting, Completed, etc.)
- 🧬 **Phases Analysis** - Distribution across different clinical trial phases
- 👥 **Demographics** - Gender and age group distributions
- 💰 **Funding Sources** - Analysis of trial funding patterns
- 🌍 **Geographic Distribution** - Top countries conducting trials
- 📅 **Temporal Trends** - Trials started by year and month
- 🔗 **Correlation Analysis** - Relationships between numerical variables
- 🔄 **Cross-tabulations** - Status vs Phases, Status vs Results, and more
- 📊 **Enrollment Statistics** - Comprehensive enrollment analysis with outlier handling

## 📁 Project Structure

```
Project 3/
│
├── COVID clinical trials.csv          # Original dataset
├── cleaned_covid_clinical_trials.csv  # Cleaned dataset (generated)
├── covid_trials_eda.py                # Main analysis script
├── README.md                           # This file
│
├── status_distribution.png             # Visualizations (generated)
├── study_results_distribution.png
├── phases_distribution.png
├── gender_distribution.png
├── age_group_distribution.png
├── enrollment_analysis.png
├── top_conditions.png
├── funding_sources.png
├── country_distribution.png
├── trials_by_year.png
├── trials_by_month.png
├── correlation_matrix.png
├── status_vs_study_results.png
└── status_vs_phases.png
```

## 🚀 Getting Started

### Prerequisites

Make sure you have the following installed:

```bash
Python 3.8 or higher
pandas
numpy
matplotlib
seaborn
pdfplumber (for PDF reading)
```

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Project 3"
   ```

2. **Install required packages**
   ```bash
   pip install pandas numpy matplotlib seaborn pdfplumber
   ```

3. **Ensure the dataset is in the project folder**
   - The dataset file `COVID clinical trials.csv` should be in the main directory

## 💻 Usage

Run the analysis script:

```bash
python covid_trials_eda.py
```

The script will:
1. 📥 Load and explore the dataset
2. 🧹 Clean and preprocess the data
3. 📊 Perform comprehensive analysis
4. 📈 Generate 14 visualizations
5. 💾 Save the cleaned dataset as `cleaned_covid_clinical_trials.csv`

## 📈 Generated Visualizations

### 1. Status Distribution 📊
Bar chart showing the distribution of clinical trial statuses.

### 2. Study Results Distribution ✅
Analysis of trials with results vs no results available.

### 3. Phases Distribution 🧪
Horizontal bar chart showing distribution across trial phases.

### 4. Gender Distribution 👥
Dual visualization (bar + pie chart) of gender distribution in trials.

### 5. Age Group Distribution 👴👶
Categorized age group analysis with simplified categories.

### 6. Enrollment Analysis 📊
Three-panel visualization:
- Histogram (outliers removed)
- Box plot
- Log-scale histogram

### 7. Top Conditions 🦠
Top 20 conditions being studied in clinical trials.

### 8. Funding Sources 💰
Top 10 funding sources with simplified categories.

### 9. Country Distribution 🌍
Top 20 countries conducting COVID-19 clinical trials.

### 10. Trials by Year 📅
Line chart showing the number of trials started each year.

### 11. Trials by Month 📆
Monthly trend analysis of trial starts over time.

### 12. Correlation Matrix 🔗
Heatmap showing correlations between numerical variables.

### 13. Status vs Study Results 🔄
Stacked bar chart showing the relationship between status and results.

### 14. Status vs Phases 🔬
Stacked bar chart showing phase distribution across different statuses.

## 📊 Key Insights

The analysis reveals:
- 🎯 Most trials are in "Recruiting" status
- 🌐 United States leads in number of trials
- 💉 Majority of trials are interventional studies
- 👥 Most trials include all gender categories
- 📈 Significant increase in trials in 2020
- 💰 "Other" is the primary funding source

## 🛠️ Technologies Used

- **Python** - Programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical visualization

## 📝 Dataset Information

- **Source**: ClinicalTrials.gov
- **Rows**: 5,783 trials
- **Columns**: 27 original columns
- **Domain**: Clinical Trials & Healthcare

## 🎨 Features of the Analysis

- ✨ Professional visualizations with proper styling
- 📊 Value labels on all charts for clarity
- 🎯 Outlier handling for skewed data
- 📈 Multiple visualization types (bar, line, pie, heatmap)
- 🌈 Color-coded charts for better readability
- 📏 Consistent formatting across all visualizations

## 📌 Notes

- All visualizations are saved as high-resolution PNG files (300 DPI)
- The cleaned dataset is saved separately for further analysis
- The script handles missing data appropriately
- Country names are extracted from location strings automatically

## 🤝 Contributing

Feel free to fork this project and submit pull requests for any improvements!

## 📧 Contact

For questions or suggestions, please open an issue in the repository.

---

**Made with ❤️ using Python and Pandas**

