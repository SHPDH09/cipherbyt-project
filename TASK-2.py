import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Data Collection
# Load the Excel dataset
file_path = r"C:\Users\SHIV\OneDrive\Desktop\Project\Data Science-2\Unemployment in India.xlsx"  # Replace with actual file path
unemployment_data = pd.read_excel(file_path)

# Step 2: Data Cleaning
unemployment_data.dropna(inplace=True)

# Convert the date column to datetime format with the correct format specification
unemployment_data['Date'] = pd.to_datetime(unemployment_data['Date'], format='%d-%m-%Y')

# Step 3: Exploratory Data Analysis (EDA)
print(unemployment_data.describe())
print(unemployment_data.info())

# Step 4: Visualization
plt.figure(figsize=(10, 6))
sns.lineplot(data=unemployment_data, x='Date', y='Estimated Unemployment Rate (%)')
plt.title('Estimated Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Estimated Unemployment Rate (%)')
plt.grid(True)
plt.show()

# Step 5: Modeling (Optional)
# For instance, using a simple linear regression model to predict future unemployment rates

# Prepare the data
unemployment_data['Year'] = unemployment_data['Date'].dt.year
X = unemployment_data[['Year']]
y = unemployment_data['Estimated Unemployment Rate (%)']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Step 6: Reporting
# Summarize findings
summary = f"""
Unemployment Analysis Report
-----------------------------
Mean Squared Error of the model: {mse}
Visualizations and further insights can be included here.
"""

print(summary)
