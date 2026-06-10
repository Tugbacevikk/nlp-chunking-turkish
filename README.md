# nlp-chunking-turkish
# Turkish Chunking with CRF

Bursa Teknik Üniversitesi — Doğal Dil İşleme Dersi Projesi  
**Konu 3: İsim ve Diğer Öbeklerin Saptanması (Chunking)**  
2025 – 2026 Bahar Dönemi
## katkı sağlayanlar 
Tuğba Çevik 223608859074
Selin Şentürk 22360859043

---

## Proje Hakkında

Bu proje, Türkçe cümleler üzerinde otomatik sözdizimsel öbek saptama (chunking) görevi için **Koşullu Rassal Alan (CRF)** modeli geliştirmektedir.

Desteklenen öbek türleri:
- **NP** — İsim Öbeği
- **VP** — Eylem Öbeği
- **ADVP** — Zarf Öbeği
- **ADJP** — Sıfat Öbeği

İki farklı model eğitilmiştir:
1. **Düz (flat) chunking** — tek katmanlı BIO etiketleme
2. **İç içe (nested) chunking** — CHUNK-OUTER, CHUNK-INNER ve CLAUSE sütunları

---

## Veri Kümesi

[UD Turkish-BOUN](https://universaldependencies.org/treebanks/tr_boun/index.html) veri kümesi kullanılmıştır.

| Küme | Cümle Sayısı |
|------|-------------|
| Train | 7.803 |
| Dev | 979 |
| Test | 979 |

---

## Kurulum

```bash
pip install sklearn-crfsuite seqeval scikit-learn matplotlib
```

---

## Çalıştırma Sırası

Tüm dosyaları aynı klasöre koy, sırasıyla çalıştır:

### 1. Düz BIO dosyaları oluştur
```bash
python convert_to_bio.py
```
→ `train_bio.txt`, `dev_bio.txt`, `test_bio.txt` üretilir.

### 2. Nested BIO dosyaları oluştur
```bash
python convert_to_bio_nested.py
```
→ `train_bio_nested.txt`, `dev_bio_nested.txt`, `test_bio_nested.txt` üretilir.

### 3. Düz modeli eğit
```bash
python train_model.py
```
→ `crf_model.pkl` kaydedilir.

### 4. Nested modeli eğit
```bash
python train_model_nested.py
```
→ `crf_outer.pkl`, `crf_inner.pkl`, `crf_clause.pkl` kaydedilir.

### 5. Tahmin yap
```bash
python predict.py           # Düz model
python predict_nested.py    # Nested model (CoNLL formatında çıktı)
```

### 6. Grafikleri oluştur
```bash
python plot_results.py
python confusion_matrix.py
python confusion_matrix_nested.py
```

---

## Sonuçlar

### Düz Model (Test Seti)

| Sınıf | Precision | Recall | F1 |
|-------|-----------|--------|----|
| ADJP | 0.80 | 0.75 | 0.77 |
| ADVP | 0.78 | 0.74 | 0.76 |
| NP | 0.80 | 0.81 | 0.80 |
| VP | 0.87 | 0.84 | 0.86 |
| **Weighted Avg** | **0.82** | **0.81** | **0.81** |

Token Accuracy: **%84.16**

### Nested Model (Test Seti)

| Model | Micro F1 | Macro F1 |
|-------|----------|----------|
| OUTER Chunk | 0.81 | 0.80 |
| INNER Chunk | 0.51 | 0.38 |
| CLAUSE Type | 0.28 | 0.30 |

---

## Dosya Yapısı

```
nlp-chunking-turkish/
│
├── convert_to_bio.py           # CoNLL-U → düz BIO dönüşümü
├── convert_to_bio_nested.py    # CoNLL-U → nested BIO dönüşümü
├── train_model.py              # Düz CRF modeli eğitimi
├── train_model_nested.py       # Nested CRF modeli eğitimi
├── predict.py                  # Düz model tahmini
├── predict_nested.py           # Nested model tahmini (CoNLL çıktısı)
├── confusion_matrix.py         # Düz model confusion matrix
├── confusion_matrix_nested.py  # Nested model confusion matrix
├── plot_results.py             # Performans grafikleri
│
├── tr_boun-ud-train.conllu     # Ham veri (train)
├── tr_boun-ud-dev.conllu       # Ham veri (dev)
├── tr_boun-ud-test.conllu      # Ham veri (test)
│
├── train_bio.txt               # Düz BIO (train)
├── dev_bio.txt                 # Düz BIO (dev)
├── test_bio.txt                # Düz BIO (test)
│
├── train_bio_nested.txt        # Nested BIO (train)
├── dev_bio_nested.txt          # Nested BIO (dev)
├── test_bio_nested.txt         # Nested BIO (test)
│
└── chunking_rapor.docx         # Proje raporu
```

---

## Kullanılan Teknolojiler

- Python 3.x
- [sklearn-crfsuite](https://sklearn-crfsuite.readthedocs.io)
- [seqeval](https://github.com/chakki-works/seqeval)
- scikit-learn
- matplotlib
