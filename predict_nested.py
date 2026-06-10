# -*- coding: utf-8 -*-
import os
import pickle

base = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(base, 'crf_outer.pkl'),  'rb') as f: crf_outer  = pickle.load(f)
with open(os.path.join(base, 'crf_inner.pkl'),  'rb') as f: crf_inner  = pickle.load(f)
with open(os.path.join(base, 'crf_clause.pkl'), 'rb') as f: crf_clause = pickle.load(f)

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

def predict_conll(sentence_str):
    tokens = sentence_str.strip().split()
    feats  = sent_features(tokens)
    outer  = crf_outer.predict([feats])[0]
    inner  = crf_inner.predict([feats])[0]
    clause = crf_clause.predict([feats])[0]

    print(f"\n# text = {sentence_str}")
    print(f"# columns = ID FORM CHUNK-OUTER CHUNK-INNER CLAUSE")
    print(f"{'ID':<4} {'FORM':<20} {'CHUNK-OUTER':<14} {'CHUNK-INNER':<14} CLAUSE")
    print("-" * 65)
    for i, (tok, o, inn, cl) in enumerate(zip(tokens, outer, inner, clause), 1):
        print(f"{i:<4} {tok:<20} {o:<14} {inn:<14} {cl}")
    print()

predict_conll("Öğrenci kütüphanede kitap okudu .")
predict_conll("Güzel bir gün dışarıda yürüyüş yaptım .")
predict_conll("Hoca derste yeni bir konu anlattı .")
predict_conll("Dün akşam toplantıdan erken çıkan öğrencinin hocasının önerdiği makaleyi kütüphanede dikkatlice okuduğunu fark ettim .")
predict_conll("Kitabi okuyan ogrenci sinava girdi .")