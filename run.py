from joblib import load
import string

def classify_message(message):
    model = load('models/multinomial_nb_model.pkl')
    tfidf_vectorizer = load('models/tfidf_vectorizer.pkl')
    
    # Preprocess the input message
    processed_message = [message.lower().translate(str.maketrans('', '', string.punctuation))]
    
    # Vectorize the input message
    vectorized_message = tfidf_vectorizer.transform(processed_message)
    
    # Predict the label (spam or ham)
    prediction = model.predict(vectorized_message)
    return prediction[0]

if __name__ == "__main__":
    # Allow user input for classification
    print("SMS Spam Detection System")
    print("=========================")
    user_message = input("Enter an SMS message to classify: ")
    prediction = classify_message(user_message)
    print(f"The message is classified as: {prediction}")
