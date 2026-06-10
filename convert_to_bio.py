import os

def read_conllu(file_path):
    sentences = []
    current_sentence = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                continue
            if line == '':
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                parts = line.split('\t')
                if len(parts) >= 8 and '-' not in parts[0] and '.' not in parts[0]:
                    token_id = parts[0]
                    form = parts[1]
                    upos = parts[3]
                    deprel = parts[7]
                    current_sentence.append((form, upos, deprel))
    
    if current_sentence:
        sentences.append(current_sentence)
    
    return sentences

def get_chunk_tag(upos, deprel):
    if upos in ['NOUN', 'PROPN', 'PRON', 'DET', 'NUM']:
        return 'NP'
    elif upos == 'VERB':
        return 'VP'
    elif upos == 'ADV':
        return 'ADVP'
    elif upos == 'ADJ':
        return 'ADJP'
    else:
        return 'O'

def convert_to_bio(sentences):
    bio_sentences = []
    
    for sentence in sentences:
        bio_sentence = []
        prev_chunk = None
        
        for form, upos, deprel in sentence:
            chunk = get_chunk_tag(upos, deprel)
            
            if chunk == 'O':
                bio_sentence.append((form, 'O'))
                prev_chunk = None
            elif chunk == prev_chunk:
                bio_sentence.append((form, f'I-{chunk}'))
            else:
                bio_sentence.append((form, f'B-{chunk}'))
                prev_chunk = chunk
        
        bio_sentences.append(bio_sentence)
    
    return bio_sentences

def save_bio(bio_sentences, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for sentence in bio_sentences:
            for form, tag in sentence:
                f.write(f'{form}\t{tag}\n')
            f.write('\n')

def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    files = {
        'train': 'tr_boun-ud-train.conllu',
        'dev':   'tr_boun-ud-dev.conllu',
        'test':  'tr_boun-ud-test.conllu'
    }
    
    for split, filename in files.items():
        input_path = os.path.join(base_path, filename)
        output_path = os.path.join(base_path, f'{split}_bio.txt')
        
        print(f'{split} okunuyor...')
        sentences = read_conllu(input_path)
        print(f'{split}: {len(sentences)} cümle bulundu')
        
        bio_sentences = convert_to_bio(sentences)
        save_bio(bio_sentences, output_path)
        print(f'{split} kaydedildi → {output_path}')
    
    print('\nTamamlandı!')

if __name__ == '__main__':
    main()