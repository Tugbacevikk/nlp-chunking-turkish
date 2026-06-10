import os
import pickle

base_path = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(base_path, 'crf_model.pkl'), 'rb') as f:
    crf = pickle.load(f)

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

def predict(sentence):
    tokens = sentence.strip().split()
    features = sentence_to_features(tokens)
    tags = crf.predict([features])[0]

    print(f"\nCümle: {sentence}")
    print("-" * 50)
    print(f"{'Token':<20} {'Etiket':<10}")
    print("-" * 50)
    for token, tag in zip(tokens, tags):
        print(f"{token:<20} {tag:<10}")
    print("-" * 50)

predict("Öğrenci kütüphanede kitap okudu .")
predict("Güzel bir gün dışarıda yürüyüş yaptım .")
predict("Hoca derste yeni bir konu anlattı .")