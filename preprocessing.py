import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def preprocessing():

    print("\n<-----------------preprocessing-------------------->")

    # Load the dataset
    data = pd.read_csv("DrugClassification.csv")

    # Delete duplicates
    data = data.drop_duplicates()

    #check for missing values
    print("\nMissing values in each column:")
    print(data.isnull().sum())

    # Check the distribution of the target variable
    print("\nDistribution of target variable:")
    print(data['Drug'].value_counts())

    print("\nData types of each column:")
    print(data.describe())

    le = LabelEncoder() 

    # Encode textual values to numerical values
    data.Sex = le.fit_transform(data.Sex)
    sex = list(le.classes_)
    data.BP = le.fit_transform(data.BP)
    bp = list(le.classes_)
    data.Cholesterol = le.fit_transform(data.Cholesterol)
    cholesterol = list(le.classes_)
    data.Drug = le.fit_transform(data.Drug)
    drug = list(le.classes_)

    # Split the dataset into features and target variable
    x = data.drop("Drug", axis=1)
    y = data.Drug

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # Print what values were encoded to what
    print("\nEncoded values:")
    print("Sex:",sex)
    print("BP:",bp)
    print("Cholesterol:", cholesterol)
    print("Drug:",drug)

    # Print the first few rows of the dataset
    print("\n", data.head())

    print("\n<-------------------------------------------------->")
    
    return data, x_train, x_test, y_train, y_test


