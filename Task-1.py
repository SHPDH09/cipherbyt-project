import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

def load_data(filepath):
    """Load the Iris dataset from an Excel file."""
    try:
        data = pd.read_excel(filepath)
        print("Columns in the dataset:", data.columns)
        # Drop the 'Id' column
        data = data.drop(columns=['Id'])
        # Rename columns to ensure consistency
        data.columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'species']
        return data
    except FileNotFoundError:
        print(f"File not found at {filepath}")
        return None

def exploratory_data_analysis(data):
    """Perform exploratory data analysis (EDA) on the dataset."""
    # Pairplot to visualize relationships between features
    sns.pairplot(data, hue='species', markers=["o", "s", "D"])
    plt.show()

    # Correlation matrix to understand relationships between features
    corr_matrix = data.drop('species', axis=1).corr()  # Exclude the 'species' column
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.show()

def preprocess_data(data):
    """Preprocess the data: split into train/test sets and standardize features."""
    X = data.drop('species', axis=1)
    y = data['species']
    
    # Convert categorical labels to numerical values
    y = pd.Categorical(y).codes

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler

def train_model(X_train, y_train):
    """Train a logistic regression model on the training data."""
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model on the test data."""
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    print(f'Accuracy: {accuracy}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Classification Report:')
    print(class_report)

    # Visualize the confusion matrix
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def make_prediction(model, scaler, new_data):
    """Make a prediction on new data using the trained model and scaler."""
    try:
        new_data_standardized = scaler.transform(new_data)
        prediction = model.predict(new_data_standardized)
        return prediction
    except Exception as e:
        print(f"Error in making prediction: {e}")
        return None

def main():
    filepath = r"C:\Users\SHIV\OneDrive\Desktop\Project\Data Science-2\Iris Flower.xlsx"
    
    # Load the dataset
    data = load_data(filepath)
    if data is not None:
        print(data.head())
        
        # Perform exploratory data analysis (EDA)
        exploratory_data_analysis(data)
        
        # Preprocess the data
        X_train, X_test, y_train, y_test, scaler = preprocess_data(data)
        
        # Train a logistic regression model
        model = train_model(X_train, y_train)
        
        # Evaluate the model
        evaluate_model(model, X_test, y_test)
        
        # Make a prediction on new data
        new_data = [[5.1, 3.5, 1.4, 0.2]]
        prediction = make_prediction(model, scaler, new_data)
        if prediction is not None:
            species = data['species'].unique()[prediction[0]]
            print(f'The predicted species is: {species}')
    else:
        print("Failed to load data.")

if __name__ == "__main__":
    main()
