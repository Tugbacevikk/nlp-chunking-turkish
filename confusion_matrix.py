import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from seqeval.metrics import classification_report

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

base_path = os.path.dirname(os.path.abspath(__file__))

print("Model yükleniyor...")
with open(os.path.join(base_path, 'crf_model.pkl'), 'rb') as f:
    crf = pickle.load(f)

test_data = read_bio_file(os.path.join(base_path, 'test_bio.txt'))
X_test = [sentence_to_features(tokens) for tokens, tags in test_data]
y_test = [tags for tokens, tags in test_data]

y_pred = crf.predict(X_test)

y_test_flat = [tag for sent in y_test for tag in sent]
y_pred_flat = [tag for sent in y_pred for tag in sent]

labels = ['B-NP', 'I-NP', 'B-VP', 'I-VP', 'B-ADVP', 'I-ADVP', 'B-ADJP', 'I-ADJP', 'O']

cm = confusion_matrix(y_test_flat, y_pred_flat, labels=labels)

fig, ax = plt.subplots(figsize=(12, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(ax=ax, cmap='Blues', colorbar=True, xticks_rotation=45)

ax.set_title('Chunking Modeli - Confusion Matrix (Test Seti)', fontsize=14, pad=20)
ax.set_xlabel('Tahmin Edilen Etiket', fontsize=12)
ax.set_ylabel('Gerçek Etiket', fontsize=12)

plt.tight_layout()
output_path = os.path.join(base_path, 'confusion_matrix.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Confusion matrix kaydedildi → {output_path}")
plt.show()

print("\n--- TOKEN BAZLI SINIFLANDIRMA RAPORU ---")
from sklearn.metrics import classification_report as skl_report
print(skl_report(y_test_flat, y_pred_flat, labels=labels))