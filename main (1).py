import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Function to read and convert data
def transpose_world_bank_data(filename):
    df = pd.read_csv(filename)
    transpose_data = df.set_index(['Country Name', 'Indicator Name']).stack().unstack(0).reset_index()
    transpose_data.columns.name = None

    return df, transpose_data


file_path = 'worldbankdata.csv'
dataframe, transpose_data = transpose_world_bank_data(file_path)
print(original_df)
print(transpose_data)

# Converted newdataframe to csv
# transpose_data.to_csv('transposeddata.csv')

# Summary Statistics
data = pd.read_csv('transposeddata.csv')
desc_stats=data.describe()
print(desc_stats)

# Correlation Heatmap
correlation_data = data[data['Indicator Name'].isin(['Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)','Urban population (% of total population)'])]
pivot_data = correlation_data.pivot(index='Year', columns='Indicator Name', values='India')
pivot_data.dropna(inplace=True)
correlation_matrix = pivot_data.corr()
plt.figure(figsize=(3, 3))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Relation between Poverty headcount and Urban areas Population - India.')
plt.show()

# Line chart

linechart_data = data[(data['Indicator Name'] == 'Annual freshwater withdrawals, total (% of internal resources)') &
                   (data['Year'].notna())]  # Ensure there's a valid year
linechart_data = data[['Year', 'Pakistan', 'Bangladesh', 'Nepal']]
grouped_data= linechart_data.groupby('Year').sum()
plt.figure(figsize=(8, 6))
plt.plot(grouped_data.index, grouped_data['Pakistan'], label='Pakistan')
plt.plot(grouped_data.index, grouped_data['Bangladesh'], label='Bangladesh')
plt.plot(grouped_data.index, grouped_data['Nepal'], label='Nepal')
plt.title('Annual Freshwater Withdrawls - Pakistan, Bangladesh, Nepal')
plt.xlabel('Year')
plt.ylabel('Annual Freshwater Withdrawls')
plt.legend()
plt.grid(True)
plt.show()

# BarChart
selected_indicator = 'Mortality rate, under-5 (per 1,000 live births)'
selected_data = dataframe[dataframe['Indicator Name'] == selected_indicator]
sum_values = selected_data.groupby('Country Name').sum().iloc[:, 2:]
top_countries = sum_values.sum(axis=1).sort_values(ascending=False).head(15)
plt.figure(figsize=(12, 6))
top_countries.plot(kind='bar', color='black')
plt.title(f'Top 15 Countries for {selected_indicator}')
plt.xlabel('Country')
plt.ylabel('Total')
plt.show()

# piechart

disaster = dataframe[dataframe['Indicator Name'] == 'Disaster risk reduction progress score (1-5 scale; 5=best)']
years_columns = dataframe.columns[2:]
disaster['Total'] = disaster[years_columns].sum(axis=1)
top5_countries = disaster.nlargest(5, 'Total')
plt.figure(figsize=(8, 8))
plt.pie(top5_countries['Total'], labels=top5_countries['Country Name'], autopct='%1.1f%%', startangle=90)
plt.legend(bbox_to_anchor=(1, 0.5), loc="center left", title="Country Name", bbox_transform=plt.gcf().transFigure)
plt.title('Less Disaster Risks')
plt.show()
