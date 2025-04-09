from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def supervised(x_train, x_test, y_train, y_test):

    print("\n<-------------------supervised--------------------->")

    # Train the DecisionTreeClassifier with different parameters
    DTmodel1 = DecisionTreeClassifier(max_depth=5).fit(x_train, y_train)
    DTmodel2 = DecisionTreeClassifier(criterion='entropy').fit(x_train, y_train)
    DTmodel3 = DecisionTreeClassifier(min_samples_split= 10).fit(x_train, y_train)

    # Train the KNeighborsClassifier with different parameters
    KNmodel1 = KNeighborsClassifier(n_neighbors=3).fit(x_train, y_train)
    KNmodel2 = KNeighborsClassifier(n_neighbors=5).fit(x_train, y_train)
    KNmodel3 = KNeighborsClassifier(n_neighbors=7, weights='distance').fit(x_train, y_train)

    # Print the accuracy of each model
    print("\nDecision Tree Classifier Accuracy:")
    print("DTmodel1 (max_depth=5):", DTmodel1.score(x_test, y_test))
    print("DTmodel2 (criterion='entropy'):", DTmodel2.score(x_test, y_test))
    print("DTmodel3 (min_samples_split=10):", DTmodel3.score(x_test, y_test))

    # Print training and testing scores for one of the DecisionTreeClassifier models
    print("\nCheck for overfitting/underfitting")
    print("Train score:", DTmodel1.score(x_train, y_train))
    print("Test score:", DTmodel1.score(x_test, y_test))

    #Make confusion matrixes
    y_pred = DTmodel1.predict(x_test)
    DTcm1 = confusion_matrix(y_test, y_pred)
    y_pred = DTmodel2.predict(x_test)
    DTcm2 = confusion_matrix(y_test, y_pred)
    y_pred = DTmodel3.predict(x_test)
    DTcm3 = confusion_matrix(y_test, y_pred)

    #Print the classification report for the DecisionTreeClassifier models
    print("\nClassification report for DTmodel1:\n", classification_report(y_test, y_pred))

    # Print the accuracy of each model
    print("\nNeighbor Classifier Accuracy:")
    print("KNmodel1 (n_neighbors=3):", KNmodel1.score(x_test, y_test))
    print("KNmodel2 (n_neighbors=5):", KNmodel2.score(x_test, y_test))
    print("KNmodel3 (n_neighbors=7, weights='distance'):", KNmodel3.score(x_test, y_test))

    # Print training and testing scores for one of the KNeighborsClassifier models
    print("\nCheck for overfitting/underfitting")
    print("Train score:", KNmodel1.score(x_train, y_train))
    print("Test score:", KNmodel1.score(x_test, y_test))

    #Make confusion matrixes
    y_pred = KNmodel1.predict(x_test)
    KNcm1 = confusion_matrix(y_test, y_pred)
    y_pred = KNmodel2.predict(x_test)
    KNcm2 = confusion_matrix(y_test, y_pred)
    y_pred = KNmodel3.predict(x_test)
    KNcm3 = confusion_matrix(y_test, y_pred)

    # Print the classification report for one of the KNeighborsClassifier models
    print("\nClassification report for KNmodel1:\n", classification_report(y_test, y_pred))


    print("\n<-------------------------------------------------->")

    return DTcm1, DTcm2, DTcm3, KNcm1, KNcm2, KNcm3