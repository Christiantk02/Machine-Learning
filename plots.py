import seaborn as sns
import matplotlib.pyplot as plt

def eda(data):

    # Print drug distribution
    data.Drug.value_counts().plot(kind='bar', color='blue', edgecolor='white')
    plt.title("Drug Distribution")
    plt.xlabel("Drug")
    plt.ylabel("Frequency")
    plt.show()

    # Print histograms of the features
    plt.figure(figsize=(10, 6)) 
    plt.subplot(1, 2, 1)
    data.Age.hist(color="blue", edgecolor="white")
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    data.Na_to_K.hist(color="red", edgecolor="white")
    plt.title("Na_to_K Distribution")
    plt.xlabel("Na_to_K")
    plt.ylabel("Frequency")
    plt.show()

    # Print boxplots of the features
    plt.figure(figsize=(10, 6)) 
    plt.subplot(1, 2, 1)
    sns.boxplot(x="Drug", y="Na_to_K", data=data)
    plt.title("Na_to_K vs Drug")
    plt.xlabel("Drug")
    plt.ylabel("Na_to_K")

    plt.subplot(1, 2, 2)
    sns.boxplot(x="Drug", y="Age", data=data)
    plt.title("Age vs Drug")
    plt.xlabel("Drug")
    plt.ylabel("Age")
    plt.show()

    # Print pairplot of the features colored by Drug
    sns.pairplot(data, hue="Drug")
    plt.suptitle("Pairplot of features colored by Drug", y=1.02)
    plt.show()


# Print the correlation matrix
def heatmap(cm, labels):

    n = len(cm)
    plt.figure(figsize=(6*n, 6))

    for i in range(n):
        plt.subplot(1, n, i+1)
        sns.heatmap(cm[i], annot=True)
        plt.title(f"Confusion Matrix for {labels[i]}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

    plt.show()


