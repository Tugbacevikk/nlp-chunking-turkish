import os
import pickle
from sklearn_crfsuite import CRF
from seqeval.metrics import classification_report
from sklearn.metrics import accuracy_score

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

def train_crf(X_tr, y_tr, label):
    print(f"\n{label} modeli eğitiliyor...")
    crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1,
              max_iterations=100, all_possible_transitions=True)
    crf.fit(X_tr, y_tr)
    print(f"{label} eğitimi tamamlandı.")
    return crf

def evaluate(crf, X, y_true, name):
    y_pred = crf.predict(X)
    print(f"\n--- {name} (Test Seti) ---")
    print(classification_report(y_true, y_pred))
    flat_true = [t for s in y_true for t in s]
    flat_pred = [t for s in y_pred for t in s]
    acc = accuracy_score(flat_true, flat_pred)
    print(f"Token Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    return y_pred

base = os.path.dirname(os.path.abspath(__file__))

print("Veriler yükleniyor...")
train_data = read_nested_bio(os.path.join(base, 'train_bio_nested.txt'))
dev_data   = read_nested_bio(os.path.join(base, 'dev_bio_nested.txt'))
test_data  = read_nested_bio(os.path.join(base, 'test_bio_nested.txt'))
print(f"Train: {len(train_data)} | Dev: {len(dev_data)} | Test: {len(test_data)}")

X_train = [sent_features(t) for t,_,_,_ in train_data]
X_dev   = [sent_features(t) for t,_,_,_ in dev_data]
X_test  = [sent_features(t) for t,_,_,_ in test_data]

y_outer_train  = [o for _,o,_,_ in train_data]
y_inner_train  = [i for _,_,i,_ in train_data]
y_clause_train = [c for _,_,_,c in train_data]
y_outer_test   = [o for _,o,_,_ in test_data]
y_inner_test   = [i for _,_,i,_ in test_data]
y_clause_test  = [c for _,_,_,c in test_data]

crf_outer  = train_crf(X_train, y_outer_train,  "OUTER")
crf_inner  = train_crf(X_train, y_inner_train,  "INNER")
crf_clause = train_crf(X_train, y_clause_train, "CLAUSE")

evaluate(crf_outer,  X_test, y_outer_test,  "OUTER CHUNK")
evaluate(crf_inner,  X_test, y_inner_test,  "INNER CHUNK")
evaluate(crf_clause, X_test, y_clause_test, "CLAUSE TYPE")

for crf, fname in [(crf_outer,  'crf_outer.pkl'),
                   (crf_inner,  'crf_inner.pkl'),
                   (crf_clause, 'crf_clause.pkl')]:
    with open(os.path.join(base, fname), 'wb') as f:
        pickle.dump(crf, f)
    print(f"Kaydedildi → {fname}")