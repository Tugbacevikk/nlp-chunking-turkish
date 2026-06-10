import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn_crfsuite import CRF
import sklearn_crfsuite
from seqeval.metrics import classification_report, f1_score
import pickle

def read_bio_file(file_path):
    sentences = []
    current_tokens = []
    current_tags = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '':
                if current_tokens:
                    sentences.append((current_tokens, current_tags))
                    current_tokens = []
                    current_tags = []
            else:
                parts = line.split('\t')
                if len(parts) == 2:
                    current_tokens.append(parts[0])
                    current_tags.append(parts[1])

    if current_tokens:
        sentences.append((current_tokens, current_tags))

    return sentences

def word_features(sentence, i):
    word = sentence[i]
    features = {
        'bias': 1.0,
        'word': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word[:3]': word[:3],
        'word[:2]': word[:2],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.length': len(word),
    }
    if i > 0:
        prev_word = sentence[i-1]
        features.update({
            '-1:word': prev_word.lower(),
            '-1:istitle': prev_word.istitle(),
            '-1:isupper': prev_word.isupper(),
            '-1:word[-3:]': prev_word[-3:],
        })
    else:
        features['BOS'] = True
    if i > 1:
        prev2_word = sentence[i-2]
        features.update({
            '-2:word': prev2_word.lower(),
            '-2:word[-3:]': prev2_word[-3:],
        })
    if i < len(sentence) - 1:
        next_word = sentence[i+1]
        features.update({
            '+1:word': next_word.lower(),
            '+1:istitle': next_word.istitle(),
            '+1:isupper': next_word.isupper(),
            '+1:word[-3:]': next_word[-3:],
        })
    else:
        features['EOS'] = True
    if i < len(sentence) - 2:
        next2_word = sentence[i+2]
        features.update({
            '+2:word': next2_word.lower(),
            '+2:word[-3:]': next2_word[-3:],
        })
    return features

def sentence_to_features(sentence):
    return [word_features(sentence, i) for i in range(len(sentence))]

def sentence_to_labels(tags):
    return tags

base_path = os.path.dirname(os.path.abspath(__file__))

print("Veriler yükleniyor...")
train_data = read_bio_file(os.path.join(base_path, 'train_bio.txt'))
dev_data   = read_bio_file(os.path.join(base_path, 'dev_bio.txt'))
test_data  = read_bio_file(os.path.join(base_path, 'test_bio.txt'))

print(f"Train: {len(train_data)} cümle")
print(f"Dev:   {len(dev_data)} cümle")
print(f"Test:  {len(test_data)} cümle")

X_train = [sentence_to_features(tokens) for tokens, tags in train_data]
y_train = [sentence_to_labels(tags)     for tokens, tags in train_data]

X_dev   = [sentence_to_features(tokens) for tokens, tags in dev_data]
y_dev   = [sentence_to_labels(tags)     for tokens, tags in dev_data]

X_test  = [sentence_to_features(tokens) for tokens, tags in test_data]
y_test  = [sentence_to_labels(tags)     for tokens, tags in test_data]

print("\nModel eğitiliyor...")
crf = CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)

crf.fit(X_train, y_train)
print("Eğitim tamamlandı!")

print("\n--- DEV SET SONUÇLARI ---")
y_pred_dev = crf.predict(X_dev)
print(classification_report(y_dev, y_pred_dev))

print("\n--- TEST SET SONUÇLARI ---")
y_pred_test = crf.predict(X_test)
print(classification_report(y_test, y_pred_test))

model_path = os.path.join(base_path, 'crf_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(crf, f)
print(f"\nModel kaydedildi → {model_path}")

from sklearn.metrics import accuracy_score
y_test_flat = [tag for sent in y_test for tag in sent]
y_pred_flat = [tag for sent in y_pred_test for tag in sent]
accuracy = accuracy_score(y_test_flat, y_pred_flat)
print(f"\n--- ACCURACY ---")
print(f"Token bazlı Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")