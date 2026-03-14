import os
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class ModelTrainer:

    def initiate_model_training(self, X_train, X_test, y_train, y_test):

        models = {

            "Logistic Regression": LogisticRegression(max_iter=1000),

            "SVM": LinearSVC(),

            "Naive Bayes": MultinomialNB(),

            "Random Forest": RandomForestClassifier()

        }

        best_model = None
        best_accuracy = 0

        for name, model in models.items():

            model.fit(X_train, y_train)

            pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, pred)

            print("\n==============================")
            print(f"MODEL : {name}")
            print("==============================")

            print("Accuracy:", accuracy)

            print("\nClassification Report:")
            print(classification_report(y_test, pred))

            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, pred))

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model

        os.makedirs("artifacts", exist_ok=True)

        pickle.dump(best_model, open("artifacts/model.pkl", "wb"))

        print("\nBest Model Saved with Accuracy:", best_accuracy)