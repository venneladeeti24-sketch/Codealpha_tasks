"""
Unemployment Analysis - CodeAlpha Task 2
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data (Unemployment Rate in India during COVID)
data = {
    'Region': ['North', 'South', 'East', 'West', 'North-East', 'North', 'South'],
    'Date': ['2020-05-31', '2020-05-31', '2020-06-30', '2020-06-30'],
    'Unemployment_Rate': [17.48, 12.5, 15.2, 18.9, 10.1, 13.2, 9.8],
    'Employment_Rate': [42.1, 45.2, 43.5, 41.0, 48.2, 44.5, 47.1]
}

df = pd.DataFrame(data)
print(df.head())
print(df.describe())

# Visualization
plt.figure(figsize=(10,5))
sns.barplot(x='Region', y='Unemployment_Rate', data=df)
plt.title('Unemployment Rate by Region - May 2020 (COVID Impact)')
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig('unemployment_plot.png')
plt.show()

# Analysis
print("\nHighest Unemployment Region:", df.loc[df['Unemployment_Rate'].idxmax()]['Region'])
print("Average Unemployment:", df['Unemployment_Rate'].mean())

# If you have real CSV, use:
# df = pd.read_csv('Unemployment in India.csv')
# Then same plots
print("\nTask 2 Completed - Unemployment Analysis Done!")
