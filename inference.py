import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

loaded_model = pickle.load(open('svc_transactionClassification', 'rb'))


def vectorizer(path):
    df = pd.read_csv(path, encoding='latin-1')
    # creating a dataframe  with just two columns
    df1 = df[['text', 'Label']].copy()

    # removing the missing values from text
    df1 = df1[pd.notnull(df1['text'])]
    df1 = df1[pd.notnull(df1['Label'])]
    df1 = df1[df1['Label'] != 'Spam']
    # creating a new column 'label id' with encoded categories
    df1['label_id'] = df1['Label'].factorize()[0]
        
    # creating a tfidf vectorizer
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5,
                            ngram_range=(1, 2),
                            stop_words='english')


    X = df1['text'] # Collection of documents
    y = df1['Label'] # Target or the labels we want to predict (i.e., the 13 different complaints of products)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.25,
                                                        random_state = 0)

    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5,
                            ngram_range=(1, 2),
                            stop_words='english')

    fitted_vectorizer = tfidf.fit(X_train)
    
    return fitted_vectorizer



if __name__ == '__main__':
    # Sample messages
    sample_messages = [
        "Dear Customer, Your A/C ###234 is debited by NPR 3000 For: 9860901204/QR Pay, KFC/ET. Never Share Password/OTP With Anyone",
        "Hi, you have received NPR 5000 from John Doe for lunch.",
        "Your credit card ending in ###1234 has been charged NPR 2000 for online shopping at XYZ Store.",
        "Your monthly phone bill of NPR 1000 is due. Please make payment to avoid service interruption.",
        "Congratulations! You've won a free vacation. Click here to claim your prize.",
    ]

    fitted_vectorizer = vectorizer('transaction_data.csv')

    # Regular expressions
    regex_debited_credited = re.compile(r'(debited|credited)')
    regex_amount = re.compile(r'(?:(?:NPR|USD|EUR)\s?\d+(?:,\d{3})*(?:\.\d+)?)')
    regex_remarks = re.compile(r'For: (.+?)(?:\.|$)')

    for i in range(len(sample_messages)):
        if loaded_model.predict(fitted_vectorizer.transform([sample_messages[i]])) == 'Financial':
            print("Transactional Message")
            match_debited_credited = regex_debited_credited.search(sample_messages[i])
            match_amount = regex_amount.search(sample_messages[i])
            match_remarks = regex_remarks.search(sample_messages[i])

            debited_credited = match_debited_credited.group(0) if match_debited_credited else None
            amount = match_amount.group(0) if match_amount else None
            remarks = match_remarks.group(1) if match_remarks else None

            print("Message:", sample_messages[i])
            print("Debited/Credited:", debited_credited)
            print("Amount:", amount)
            print("Remarks:", remarks)
        