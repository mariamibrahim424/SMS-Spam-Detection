from sklearn.model_selection import train_test_split
from src.data_preprocessing import pre_process_data
from src.data_balancing import balance_data_undersample
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib 

def train_multinomial_nb(X_train, y_train):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

def save_model(model, filepath):
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model

def train_model():
    data = pre_process_data('./data/raw_sms.csv')
    balanced_data = balance_data_undersample(data)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(balanced_data['balanced_sms'])

    X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, balanced_data['label'], test_size=0.2, random_state=42)
    joblib.dump(vectorizer, './models/tfidf_vectorizer.pkl')
    model = train_multinomial_nb(X_train, y_train)

    evaluate_model(model, X_test, y_test)
    joblib.dump(model, './models/multinomial_nb_model.pkl')

if __name__ == "__main__":
    train_model()  # Start the training process