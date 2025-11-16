"""
COVID-19 Clinical Trials EDA
Comprehensive Exploratory Data Analysis using Pandas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("="*80)
print("COVID-19 CLINICAL TRIALS - EXPLORATORY DATA ANALYSIS")
print("="*80)

# ============================================================================
# 1. DATA LOADING
# ============================================================================
print("\n1. LOADING DATA...")
print("-" * 80)

df = pd.read_csv('COVID clinical trials.csv')
print(f"[OK] Data loaded successfully!")
print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ============================================================================
# 2. INITIAL DATA EXPLORATION
# ============================================================================
print("\n2. INITIAL DATA EXPLORATION")
print("-" * 80)

print("\n2.1 Column Names:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

print("\n2.2 First Few Rows:")
print(df.head())

print("\n2.3 Data Types:")
print(df.dtypes)

print("\n2.4 Basic Info:")
print(df.info())

print("\n2.5 Missing Values:")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing Percentage': missing_pct
})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
print(missing_df)

# Missing values analysis displayed above

# ============================================================================
# 3. DESCRIPTIVE STATISTICS
# ============================================================================
print("\n3. DESCRIPTIVE STATISTICS")
print("-" * 80)

print("\n3.1 Numerical Columns Summary:")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if numeric_cols:
    print(df[numeric_cols].describe())

print("\n3.2 Categorical Columns Summary:")
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"  Found {len(categorical_cols)} categorical columns")

# ============================================================================
# 4. DATA CLEANING
# ============================================================================
print("\n4. DATA CLEANING")
print("-" * 80)

# Create a copy for cleaning
df_clean = df.copy()

# Check missing data percentage before cleaning
print("\n4.1 Missing Data Percentage:")
missing_pct_before = (df_clean.isnull().sum() / len(df_clean)) * 100
print(missing_pct_before.sort_values(ascending=False))

# 4.2 Drop columns with high percentage of missing values (as per PDF)
# Results First Posted: 99.3%, Study Documents: 96.8% (too high to impute)
columns_to_drop = []
if 'Results First Posted' in df_clean.columns:
    if missing_pct_before['Results First Posted'] > 95:
        columns_to_drop.append('Results First Posted')
        print(f"\n[INFO] Dropping 'Results First Posted' ({missing_pct_before['Results First Posted']:.2f}% missing)")

if 'Study Documents' in df_clean.columns:
    if missing_pct_before['Study Documents'] > 95:
        columns_to_drop.append('Study Documents')
        print(f"[INFO] Dropping 'Study Documents' ({missing_pct_before['Study Documents']:.2f}% missing)")

if columns_to_drop:
    df_clean = df_clean.drop(columns=columns_to_drop)
    print(f"[OK] Dropped {len(columns_to_drop)} columns with high missing values")

# 4.3 Drop duplicate rows
print(f"\n4.3 Checking for duplicates:")
print(f"  Shape before dropping duplicates: {df_clean.shape}")
df_clean = df_clean.drop_duplicates()
print(f"  Shape after dropping duplicates: {df_clean.shape}")

# 4.4 Drop rows with less than 10 non-null values (as per PDF)
print(f"\n4.4 Dropping rows with less than 10 non-null values:")
print(f"  Shape before: {df_clean.shape}")
df_clean = df_clean.dropna(axis=0, thresh=10)
print(f"  Shape after: {df_clean.shape}")

# 4.5 Extract Country from Locations (as per PDF)
if 'Locations' in df_clean.columns:
    print("\n4.5 Extracting Country from Locations:")
    countries = []
    for i in range(df_clean.shape[0]):
        loc = str(df_clean.Locations.iloc[i])
        if loc != 'nan':
            loc_parts = loc.split(',')
            country = loc_parts[-1].strip() if loc_parts else 'Unknown'
        else:
            country = 'Unknown'
        countries.append(country)
    df_clean['Country'] = countries
    print(f"[OK] Country column extracted")
    print(f"  Top 10 countries: {df_clean['Country'].value_counts().head(10).to_dict()}")

# Clean date columns
date_columns = ['Start Date', 'Primary Completion Date', 'Completion Date', 
                'First Posted', 'Last Update Posted']

for col in date_columns:
    if col in df_clean.columns:
        df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')

# Clean numeric columns
if 'Enrollment' in df_clean.columns:
    df_clean['Enrollment'] = pd.to_numeric(df_clean['Enrollment'], errors='coerce')

print("\n[OK] Date columns converted to datetime")
print("[OK] Numeric columns cleaned")

# ============================================================================
# 5. ANALYSIS OF KEY VARIABLES
# ============================================================================
print("\n5. ANALYSIS OF KEY VARIABLES")
print("-" * 80)

# 5.1 Status Analysis
if 'Status' in df_clean.columns:
    print("\n5.1 Status Distribution:")
    status_counts = df_clean['Status'].value_counts()
    print(status_counts)
    
    # Visualization
    plt.figure(figsize=(14, 8))
    bars = status_counts.plot(kind='bar', color='steelblue', edgecolor='black', linewidth=1.2)
    plt.title('Distribution of Clinical Trial Status', fontsize=16, fontweight='bold')
    plt.xlabel('Status', fontsize=12)
    plt.ylabel('Number of Trials', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (idx, val) in enumerate(status_counts.items()):
        plt.text(i, val + val*0.01, f'{val:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('status_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: status_distribution.png")

# 5.2 Study Results Analysis
if 'Study Results' in df_clean.columns:
    print("\n5.2 Study Results Distribution:")
    results_counts = df_clean['Study Results'].value_counts()
    print(results_counts)
    
    # Visualization - Use bar chart for better readability (highly imbalanced data)
    plt.figure(figsize=(10, 8))
    colors = ['#ff6b6b', '#51cf66']
    bars = results_counts.plot(kind='bar', color=colors, edgecolor='black', linewidth=1.5)
    plt.title('Distribution of Study Results', fontsize=16, fontweight='bold')
    plt.xlabel('Study Results', fontsize=12)
    plt.ylabel('Number of Trials', fontsize=12)
    plt.xticks(rotation=0, ha='center')
    
    # Add value labels on bars
    for i, (idx, val) in enumerate(results_counts.items()):
        plt.text(i, val + val*0.01, f'{val:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        plt.text(i, val/2, f'{val/len(df_clean)*100:.1f}%', ha='center', va='center', 
                fontsize=10, fontweight='bold', color='white')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('study_results_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: study_results_distribution.png")

# 5.3 Phases Analysis
if 'Phases' in df_clean.columns:
    print("\n5.3 Clinical Trial Phases Distribution:")
    phases_counts = df_clean['Phases'].value_counts()
    print(phases_counts)
    
    # Visualization
    plt.figure(figsize=(12, 8))
    bars = phases_counts.plot(kind='barh', color='coral', edgecolor='black', linewidth=1.2)
    plt.title('Distribution of Clinical Trial Phases', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Trials', fontsize=12)
    plt.ylabel('Phase', fontsize=12)
    plt.gca().invert_yaxis()  # Highest at top
    
    # Add value labels on bars
    for i, (idx, val) in enumerate(phases_counts.items()):
        plt.text(val + val*0.01, i, f'{val:,}', va='center', fontsize=10, fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('phases_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: phases_distribution.png")

# 5.4 Study Type Analysis
if 'Study Type' in df_clean.columns:
    print("\n5.4 Study Type Distribution:")
    study_type_counts = df_clean['Study Type'].value_counts()
    print(study_type_counts)

# 5.5 Gender Distribution
if 'Gender' in df_clean.columns:
    print("\n5.5 Gender Distribution:")
    gender_counts = df_clean['Gender'].value_counts()
    print(gender_counts)
    
    # Visualization - Use both bar chart and pie chart
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Bar chart
    colors_gender = ['#4ecdc4', '#ff6b6b', '#95e1d3']
    bars = gender_counts.plot(kind='bar', ax=axes[0], color=colors_gender, edgecolor='black', linewidth=1.5)
    axes[0].set_title('Gender Distribution in Clinical Trials', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Gender', fontsize=12)
    axes[0].set_ylabel('Number of Trials', fontsize=12)
    axes[0].tick_params(axis='x', rotation=0)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (idx, val) in enumerate(gender_counts.items()):
        axes[0].text(i, val + val*0.01, f'{val:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Pie chart
    gender_counts.plot(kind='pie', ax=axes[1], autopct='%1.1f%%', startangle=90,
                      colors=colors_gender, textprops={'fontsize': 12, 'fontweight': 'bold'})
    axes[1].set_title('Gender Distribution (Percentage)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('')
    
    plt.tight_layout()
    plt.savefig('gender_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: gender_distribution.png")

# 5.6 Age Group Analysis (as per PDF)
if 'Age' in df_clean.columns:
    print("\n5.6 Age Group Distribution:")
    
    # Create simplified age categories
    def categorize_age(age_str):
        if pd.isna(age_str):
            return 'Unknown'
        age_lower = str(age_str).lower()
        
        # Child-only categories
        if ('child' in age_lower and 'adult' not in age_lower) or \
           ('up to' in age_lower) or \
           any(x in age_lower for x in ['month', 'neonatal', 'infant', 'newborn']):
            if 'year' in age_lower and 'child' in age_lower:
                # Extract age range for children
                numbers = re.findall(r'\d+', age_lower)
                if numbers:
                    max_age = int(numbers[-1])
                    if max_age < 18:
                        return 'Child Only'
            return 'Child Only'
        
        # All ages (child + adult + older adult)
        if 'child' in age_lower and 'adult' in age_lower and 'older adult' in age_lower:
            return 'All Ages (Child, Adult, Older Adult)'
        
        # Adults & Older Adults (no specific age range)
        if ('adult' in age_lower and 'older adult' in age_lower) and '18' not in age_lower:
            return 'Adults & Older Adults'
        
        # Age range categories (18+ with upper limit)
        if '18' in age_lower and ('to' in age_lower or 'years' in age_lower):
            numbers = re.findall(r'\d+', age_lower)
            if numbers and len(numbers) >= 2:
                min_age = int(numbers[0])
                max_age = int(numbers[1])
                if min_age == 18:
                    # Group by ranges
                    if max_age >= 99 or max_age >= 80:
                        return '18+ Years (All Adults)'
                    elif max_age >= 75:
                        return '18-75 Years'
                    elif max_age >= 65:
                        return '18-65 Years'
                    elif max_age >= 60:
                        return '18-60 Years'
                    elif max_age >= 55:
                        return '18-55 Years'
                    elif max_age >= 50:
                        return '18-50 Years'
                    elif max_age >= 45:
                        return '18-45 Years'
                    elif max_age >= 40:
                        return '18-40 Years'
                    elif max_age >= 35:
                        return '18-35 Years'
                    elif max_age >= 30:
                        return '18-30 Years'
                    else:
                        return '18-30 Years (Young Adults)'
        
        # 18 and older (no upper limit)
        if '18' in age_lower and 'older' in age_lower:
            return '18+ Years (All Adults)'
        
        # Fallback: return a simplified version
        return 'Other Age Groups'
    
    # Apply categorization
    df_clean['Age_Category'] = df_clean['Age'].apply(categorize_age)
    
    # Count simplified categories
    age_counts = df_clean['Age_Category'].value_counts()
    print("Age Category Distribution:")
    print(age_counts)
    
    # Visualization - Show top 10-12 categories for better readability
    plt.figure(figsize=(14, 8))
    # Show categories with at least 20 trials, or top 12, whichever is more
    min_count = 20
    display_counts = age_counts[age_counts >= min_count] if len(age_counts[age_counts >= min_count]) >= 8 else age_counts.head(12)
    
    display_counts.plot(kind='barh', color='orange', edgecolor='darkorange', linewidth=1.2)
    plt.title('Age Group Distribution in Clinical Trials', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Trials', fontsize=12)
    plt.ylabel('Age Category', fontsize=12)
    plt.gca().invert_yaxis()  # Show highest at top
    plt.tight_layout()
    plt.savefig('age_group_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: age_group_distribution.png")

# 5.7 Enrollment Analysis
if 'Enrollment' in df_clean.columns:
    print("\n5.7 Enrollment Statistics:")
    enrollment_stats = df_clean['Enrollment'].describe()
    print(enrollment_stats)
    
    # Remove extreme outliers for better visualization (use IQR method)
    Q1 = df_clean['Enrollment'].quantile(0.25)
    Q3 = df_clean['Enrollment'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter for visualization (keep outliers in stats but remove for plot)
    enrollment_filtered = df_clean[(df_clean['Enrollment'] >= lower_bound) & 
                                    (df_clean['Enrollment'] <= upper_bound)]['Enrollment']
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Histogram (without extreme outliers)
    axes[0].hist(enrollment_filtered, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].set_title('Enrollment Distribution (Outliers Removed)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Enrollment', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Box plot (with all data but capped at reasonable upper limit)
    enrollment_for_box = df_clean['Enrollment'].copy()
    # Cap extreme values at 99th percentile for box plot visualization
    p99 = enrollment_for_box.quantile(0.99)
    enrollment_for_box[enrollment_for_box > p99] = p99
    axes[1].boxplot(enrollment_for_box.dropna(), vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7))
    axes[1].set_title('Enrollment Box Plot', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Enrollment', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # Log scale histogram to show all data
    enrollment_log = df_clean['Enrollment'][df_clean['Enrollment'] > 0].copy()
    axes[2].hist(np.log10(enrollment_log), bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[2].set_title('Enrollment Distribution (Log Scale)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Log10(Enrollment)', fontsize=12)
    axes[2].set_ylabel('Frequency', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('enrollment_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: enrollment_analysis.png")

# 5.8 Top Conditions
if 'Conditions' in df_clean.columns:
    print("\n5.8 Top 20 Conditions:")
    # Split conditions by | if multiple
    all_conditions = df_clean['Conditions'].dropna().str.split('|').explode().str.strip()
    top_conditions = all_conditions.value_counts().head(20)
    print(top_conditions)
    
    # Visualization
    plt.figure(figsize=(14, 10))
    bars = top_conditions.plot(kind='barh', color='teal', edgecolor='black', linewidth=1.2)
    plt.title('Top 20 Conditions in Clinical Trials', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Trials', fontsize=12)
    plt.ylabel('Condition', fontsize=12)
    plt.gca().invert_yaxis()  # Highest at top
    
    # Add value labels on bars
    for i, (idx, val) in enumerate(top_conditions.items()):
        plt.text(val + val*0.01, i, f'{val:,}', va='center', fontsize=9, fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('top_conditions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: top_conditions.png")

# 5.9 Top Interventions
if 'Interventions' in df_clean.columns:
    print("\n5.9 Top 20 Interventions:")
    # Split interventions by | if multiple
    all_interventions = df_clean['Interventions'].dropna().str.split('|').explode().str.strip()
    top_interventions = all_interventions.value_counts().head(20)
    print(top_interventions)

# 5.10 Funded By Analysis
if 'Funded Bys' in df_clean.columns:
    print("\n5.10 Funding Sources Distribution:")
    funded_by_counts = df_clean['Funded Bys'].value_counts()
    print(funded_by_counts)
    
    # Simplify funding categories - group similar ones
    def simplify_funding(fund_str):
        if pd.isna(fund_str):
            return 'Unknown'
        fund_lower = str(fund_str).lower()
        
        # Pure categories
        if fund_lower == 'other':
            return 'Other'
        elif fund_lower == 'industry':
            return 'Industry'
        elif fund_lower == 'nih':
            return 'NIH'
        elif fund_lower == 'u.s. fed':
            return 'U.S. Fed'
        
        # Combined categories - simplify
        if 'other' in fund_lower and 'industry' in fund_lower:
            return 'Other + Industry'
        elif 'industry' in fund_lower and 'other' in fund_lower:
            return 'Other + Industry'
        elif 'other' in fund_lower and 'nih' in fund_lower:
            return 'Other + NIH'
        elif 'other' in fund_lower and 'u.s. fed' in fund_lower:
            return 'Other + U.S. Fed'
        elif 'industry' in fund_lower and 'nih' in fund_lower:
            return 'Industry + NIH'
        elif 'industry' in fund_lower and 'u.s. fed' in fund_lower:
            return 'Industry + U.S. Fed'
        else:
            return 'Other Combinations'
    
    # Create simplified funding categories
    df_clean['Funding_Simplified'] = df_clean['Funded Bys'].apply(simplify_funding)
    funding_simplified = df_clean['Funding_Simplified'].value_counts()
    
    print("\nSimplified Funding Categories:")
    print(funding_simplified)
    
    # Visualization - Show top 10 categories in a bar chart
    plt.figure(figsize=(14, 8))
    top_funding = funding_simplified.head(10)
    colors_funding = plt.cm.Set3(range(len(top_funding)))
    bars = top_funding.plot(kind='barh', color=colors_funding, edgecolor='black', linewidth=1.2)
    plt.title('Top 10 Funding Sources Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Trials', fontsize=12)
    plt.ylabel('Funding Source', fontsize=12)
    plt.gca().invert_yaxis()  # Highest at top
    
    # Add value labels
    for i, (idx, val) in enumerate(top_funding.items()):
        plt.text(val + val*0.01, i, f'{val:,} ({val/len(df_clean)*100:.1f}%)', 
                va='center', fontsize=10, fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('funding_sources.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: funding_sources.png")

# 5.11 Country Distribution (from extracted Country column)
if 'Country' in df_clean.columns:
    print("\n5.11 Country Distribution:")
    country_counts = df_clean['Country'].value_counts().head(35)
    print(country_counts)
    
    # Visualization - Top 20 countries
    top_countries = country_counts.head(20)
    plt.figure(figsize=(14, 10))
    bars = top_countries.plot(kind='barh', color='purple', edgecolor='black', linewidth=1.2)
    plt.title('Top 20 Countries in Clinical Trials', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Trials', fontsize=12)
    plt.ylabel('Country', fontsize=12)
    plt.gca().invert_yaxis()  # Highest at top
    
    # Add value labels on bars
    for i, (idx, val) in enumerate(top_countries.items()):
        plt.text(val + val*0.01, i, f'{val:,}', va='center', fontsize=9, fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('country_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: country_distribution.png")

# ============================================================================
# 6. TEMPORAL ANALYSIS
# ============================================================================
print("\n6. TEMPORAL ANALYSIS")
print("-" * 80)

# 6.1 Trials Over Time
if 'Start Date' in df_clean.columns:
    df_clean['Year'] = df_clean['Start Date'].dt.year
    df_clean['Month'] = df_clean['Start Date'].dt.month
    
    print("\n6.1 Trials Started by Year:")
    trials_by_year = df_clean['Year'].value_counts().sort_index()
    print(trials_by_year)
    
    # Visualization
    plt.figure(figsize=(14, 8))
    trials_by_year.plot(kind='line', marker='o', linewidth=2.5, markersize=10, color='darkblue', markerfacecolor='lightblue', markeredgecolor='darkblue', markeredgewidth=2)
    plt.title('Number of Clinical Trials Started by Year', fontsize=16, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Trials', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add value labels on markers
    for year, count in trials_by_year.items():
        if pd.notna(year) and pd.notna(count):
            plt.text(year, count, f' {count:,}', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('trials_by_year.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: trials_by_year.png")

# 6.2 Trials Started by Month (as per PDF)
if 'Start Date' in df_clean.columns:
    print("\n6.2 Trials Started by Month:")
    # Convert to period and count
    df_clean['Start_Month'] = df_clean['Start Date'].dt.to_period('M')
    trials_by_month = df_clean['Start_Month'].value_counts().sort_index()
    print(trials_by_month)
    
    # Visualization
    plt.figure(figsize=(16, 8))
    trials_by_month.plot(kind='line', marker='o', linewidth=2.5, markersize=6, color='darkgreen', markerfacecolor='lightgreen', markeredgecolor='darkgreen', markeredgewidth=1.5)
    plt.title('Number of Clinical Trials Started Over Time (by Month)', fontsize=16, fontweight='bold')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Number of Trials', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add value labels for top 10 months
    top_months = trials_by_month.nlargest(10)
    for month, count in top_months.items():
        if pd.notna(month) and pd.notna(count):
            plt.text(month, count, f' {count:,}', va='bottom', fontsize=8, fontweight='bold', rotation=45)
    
    plt.tight_layout()
    plt.savefig('trials_by_month.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: trials_by_month.png")

# ============================================================================
# 7. CORRELATION ANALYSIS
# ============================================================================
print("\n7. CORRELATION ANALYSIS")
print("-" * 80)

if len(numeric_cols) > 1:
    print("\n7.1 Correlation Matrix:")
    corr_matrix = df_clean[numeric_cols].corr()
    print(corr_matrix)
    
    # Visualization
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix of Numerical Variables', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: correlation_matrix.png")

# ============================================================================
# 8. CROSS-TABULATION ANALYSIS
# ============================================================================
print("\n8. CROSS-TABULATION ANALYSIS")
print("-" * 80)

# 8.1 Status vs Study Results
if 'Status' in df_clean.columns and 'Study Results' in df_clean.columns:
    print("\n8.1 Status vs Study Results:")
    crosstab1 = pd.crosstab(df_clean['Status'], df_clean['Study Results'], margins=True)
    print(crosstab1)
    
    # Visualization
    crosstab_plot = pd.crosstab(df_clean['Status'], df_clean['Study Results'], margins=False)
    plt.figure(figsize=(14, 8))
    crosstab_plot.plot(kind='bar', stacked=True, colormap='viridis', edgecolor='black', linewidth=1.2)
    plt.title('Status vs Study Results Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Status', fontsize=12)
    plt.ylabel('Number of Trials', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Study Results', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('status_vs_study_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: status_vs_study_results.png")

# 8.2 Status vs Phases (as per PDF - specifically mentioned)
if 'Status' in df_clean.columns and 'Phases' in df_clean.columns:
    print("\n8.2 Status vs Phases:")
    crosstab2 = pd.crosstab(df_clean['Status'], df_clean['Phases'], margins=True)
    print(crosstab2)
    
    # Visualization
    crosstab2_plot = pd.crosstab(df_clean['Status'], df_clean['Phases'], margins=False)
    plt.figure(figsize=(16, 8))
    crosstab2_plot.plot(kind='bar', stacked=True, colormap='Set3', edgecolor='black', linewidth=1.2)
    plt.title('Status vs Phases Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Status', fontsize=12)
    plt.ylabel('Number of Trials', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Phases', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('status_vs_phases.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: status_vs_phases.png")

# 8.3 Conditions vs Outcome Measures (as per PDF)
if 'Conditions' in df_clean.columns and 'Outcome Measures' in df_clean.columns:
    print("\n8.3 Conditions vs Outcome Measures:")
    # Group by Conditions and aggregate Outcome Measures
    conditions_outcomes = df_clean.groupby('Conditions')['Outcome Measures'].apply(
        lambda x: ' | '.join(x.dropna().astype(str)) if x.notna().any() else 'No Data'
    ).reset_index()
    conditions_outcomes.columns = ['Condition', 'Outcome Measures']
    
    # Show top 20 conditions with their outcome measures
    top_conditions_list = df_clean['Conditions'].value_counts().head(20).index.tolist()
    conditions_outcomes_top = conditions_outcomes[conditions_outcomes['Condition'].isin(top_conditions_list)]
    
    print(f"  Analyzing top {len(conditions_outcomes_top)} conditions...")
    print(f"  Sample (first 5):")
    print(conditions_outcomes_top.head())

# 8.4 Study Type vs Phases
if 'Study Type' in df_clean.columns and 'Phases' in df_clean.columns:
    print("\n8.4 Study Type vs Phases:")
    crosstab3 = pd.crosstab(df_clean['Study Type'], df_clean['Phases'], margins=True)
    print(crosstab3)

# ============================================================================
# 8.5 Saving Cleaned Data (as per PDF)
# ============================================================================
print("\n8.5 SAVING CLEANED DATA")
print("-" * 80)
df_clean.to_csv('cleaned_covid_clinical_trials.csv', index=False)
print(f"[OK] Saved cleaned dataset: cleaned_covid_clinical_trials.csv")
print(f"  Shape: {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")

# ============================================================================
# 9. SUMMARY STATISTICS
# ============================================================================
print("\n9. SUMMARY STATISTICS")
print("-" * 80)

summary_stats = {
    'Total Trials': len(df_clean),
    'Total Columns': len(df_clean.columns),
    'Trials with Results': df_clean['Study Results'].notna().sum() if 'Study Results' in df_clean.columns else 0,
    'Total Enrollment': df_clean['Enrollment'].sum() if 'Enrollment' in df_clean.columns else 0,
    'Average Enrollment': df_clean['Enrollment'].mean() if 'Enrollment' in df_clean.columns else 0,
    'Median Enrollment': df_clean['Enrollment'].median() if 'Enrollment' in df_clean.columns else 0,
}

print("\nOverall Summary:")
for key, value in summary_stats.items():
    if isinstance(value, float):
        print(f"  {key}: {value:,.2f}")
    else:
        print(f"  {key}: {value:,}")

# Summary statistics displayed above

# ============================================================================
# 10. DATA QUALITY REPORT
# ============================================================================
print("\n10. DATA QUALITY REPORT")
print("-" * 80)

quality_report = {
    'Column': [],
    'Total Values': [],
    'Non-Null Values': [],
    'Null Values': [],
    'Null Percentage': [],
    'Unique Values': [],
    'Data Type': []
}

for col in df_clean.columns:
    quality_report['Column'].append(col)
    quality_report['Total Values'].append(len(df_clean))
    quality_report['Non-Null Values'].append(df_clean[col].notna().sum())
    quality_report['Null Values'].append(df_clean[col].isna().sum())
    quality_report['Null Percentage'].append((df_clean[col].isna().sum() / len(df_clean)) * 100)
    quality_report['Unique Values'].append(df_clean[col].nunique())
    quality_report['Data Type'].append(str(df_clean[col].dtype))

quality_df = pd.DataFrame(quality_report)
print(quality_df)

# ============================================================================
# COMPLETION
# ============================================================================
print("\n" + "="*80)
print("EDA ANALYSIS COMPLETED SUCCESSFULLY!")
print("="*80)
print("\nAll output files have been saved in the current directory:")
print("  • PNG files with visualizations")
print("  • Cleaned dataset CSV file")
print("\nGenerated Files:")
print("  Visualizations (PNG):")
print("  - status_distribution.png")
print("  - study_results_distribution.png")
print("  - phases_distribution.png")
print("  - gender_distribution.png")
print("  - age_group_distribution.png")
print("  - enrollment_analysis.png")
print("  - top_conditions.png")
print("  - funding_sources.png")
print("  - country_distribution.png")
print("  - trials_by_year.png")
print("  - trials_by_month.png")
print("  - correlation_matrix.png")
print("  - status_vs_study_results.png")
print("  - status_vs_phases.png")
print("  Data:")
print("  - cleaned_covid_clinical_trials.csv")
print("\n" + "="*80)

