import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import seaborn as sns

# Read the Excel file into a pandas DataFrame
file_path = 'DatosTrabajo1.xls'
df = pd.read_excel(file_path)

df['Grasas_sat'].replace(999.99, df['Grasas_sat'].mean(), inplace=True)
df['Alcohol'].replace(999.99, df['Alcohol'].mean(), inplace=True)

#df.to_excel('output_file.xlsx', index=False)

# Set the style for the plots
sns.set(style="whitegrid")

# Create separate boxplots
#df.boxplot(column=['Grasas_sat'])
#df.boxplot(column=['Alcohol'])
#df.boxplot(column=['Calorías'])

# Create side-by-side boxplots using seaborn
sns.boxplot(x='Sexo', y='Calorías', data=df, width=0.3, palette="Set3", showfliers = True)

# Add a title to the plot
plt.title('Boxplots of Calorías')

# Display the plot
plt.show()
