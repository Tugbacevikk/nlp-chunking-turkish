import os
import pickle
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

base = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(base, 'crf_outer.pkl'),  'rb') as f: crf_outer  = pickle.load(f)
with open(os.path.join(base, 'crf_inner.pkl'),  'rb') as f: crf_inner  = pickle.load(f)
with open(os.path.join(base, 'crf_clause.pkl'), 'rb') as f: crf_clause = pickle.load(f)

def read_nested_bio(file_path):
    sentences = []
    tokens, outer, inner, clause = [], [], [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '':
                if tokens:
                    sentences.append((tokens, outer, inner, clause))
                    tokens, outer, inner, clause = [], [], [], []
            else:
                parts = line.split('\t')
                if len(parts) == 4:
                    tokens.append(parts[0]); outer.append(parts[1])
                    inner.append(parts[2]);  clause.append(parts[3])
    if tokens:
        sentences.append((tokens, outer, inner, clause))
    return sentences

def word_features(sentence, i):
    word = sentence[i]
    features = {
        'bias': 1.0, 'word': word.lower(),
        'word[-3:]': word[-3:], 'word[-2:]': word[-2:],
        'word[:3]': word[:3],   'word[:2]': word[:2],
        'word.isupper()': word.isupper(), 'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(), 'word.length': len(word),
    }
    if i > 0:
        pw = sentence[i-1]
        features.update({'-1:word': pw.lower(), '-1:istitle': pw.istitle(),
                         '-1:isupper': pw.isupper(), '-1:word[-3:]': pw[-3:]})
    else:
        features['BOS'] = True
    if i > 1:
        pw2 = sentence[i-2]
        features.update({'-2:word': pw2.lower(), '-2:word[-3:]': pw2[-3:]})
    if i < len(sentence) - 1:
        nw = sentence[i+1]
        features.update({'+1:word': nw.lower(), '+1:istitle': nw.istitle(),
                         '+1:isupper': nw.isupper(), '+1:word[-3:]': nw[-3:]})
    else:
        features['EOS'] = True
    if i < len(sentence) - 2:
        nw2 = sentence[i+2]
        features.update({'+2:word': nw2.lower(), '+2:word[-3:]': nw2[-3:]})
    return features

def sent_features(tokens):
    return [word_features(tokens, i) for i in range(len(tokens))]

test_data = read_nested_bio(os.path.join(base, 'test_bio_nested.txt'))
X_test = [sent_features(t) for t,_,_,_ in test_data]

y_outer_test  = [o for _,o,_,_ in test_data]
y_inner_test  = [i for _,_,i,_ in test_data]
y_clause_test = [c for _,_,_,c in test_data]

pred_outer  = crf_outer.predict(X_test)
pred_inner  = crf_inner.predict(X_test)
pred_clause = crf_clause.predict(X_test)

def plot_cm(y_true_list, y_pred_list, labels, title, out_path):
    y_true = [t for s in y_true_list for t in s]
    y_pred = [t for s in y_pred_list for t in s]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap='Blues', colorbar=True, xticks_rotation=45)
    ax.set_title(title, fontsize=13, pad=20)
    ax.set_xlabel('Tahmin Edilen Etiket', fontsize=11)
    ax.set_ylabel('Gerçek Etiket', fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Kaydedildi: {out_path}')

plot_cm(y_outer_test, pred_outer,
        ['B-NP','I-NP','B-VP','I-VP','B-ADVP','I-ADVP','B-ADJP','I-ADJP','O'],
        'Nested - OUTER Chunk Confusion Matrix (Test Seti)',
        os.path.join(base, 'cm_outer.png'))

plot_cm(y_inner_test, pred_inner,
        ['B-NP','I-NP','B-VP','I-VP','B-ADVP','I-ADVP','B-ADJP','I-ADJP','O'],
        'Nested - INNER Chunk Confusion Matrix (Test Seti)',
        os.path.join(base, 'cm_inner.png'))

plot_cm(y_clause_test, pred_clause,
        ['B-RELCL','I-RELCL','B-COMPCL','I-COMPCL','O'],
        'Nested - CLAUSE Type Confusion Matrix (Test Seti)',
        os.path.join(base, 'cm_clause.png'))