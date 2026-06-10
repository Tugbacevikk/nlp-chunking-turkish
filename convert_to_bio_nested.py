import os

RELCL_DEPRELS  = {'acl', 'acl:relcl'}
COMPCL_DEPRELS = {'ccomp', 'xcomp', 'advcl'}

def read_conllu(file_path):
    sentences = []
    current = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('#'):
                continue
            if line.strip() == '':
                if current:
                    sentences.append(current)
                    current = []
                continue
            parts = line.split('\t')
            if len(parts) < 8:
                continue
            tid = parts[0]
            if '-' in tid or '.' in tid:
                continue
            form, upos, deprel, head_id = parts[1], parts[3], parts[7], parts[6]
            current.append((int(tid), form, upos, deprel, head_id))
    if current:
        sentences.append(current)
    return sentences

OUTER_UPOS_MAP = {
    'NOUN': 'NP', 'PROPN': 'NP', 'PRON': 'NP', 'DET': 'NP', 'NUM': 'NP',
    'VERB': 'VP', 'ADV': 'ADVP', 'ADJ': 'ADJP',
}

def outer_chunk(upos):
    return OUTER_UPOS_MAP.get(upos, 'O')

def get_clause_type(deprel):
    if deprel in RELCL_DEPRELS:  return 'RELCL'
    if deprel in COMPCL_DEPRELS: return 'COMPCL'
    return None

def build_clause_spans(tokens):
    clause_label = {t[0]: None for t in tokens}
    for tid, form, upos, deprel, head_id in tokens:
        ctype = get_clause_type(deprel)
        if ctype is None:
            continue
        members = [tid] + [t2id for t2id,_,_,_,t2head in tokens if t2head == str(tid)]
        members.sort()
        for m in members:
            if clause_label[m] is None:
                clause_label[m] = ctype
    return clause_label

def convert_sentence(tokens):
    outer_map = {}
    prev_outer = None
    for tid, form, upos, deprel, head_id in tokens:
        chunk = outer_chunk(upos)
        if chunk == 'O':
            outer_map[tid] = 'O'; prev_outer = None
        elif chunk == prev_outer:
            outer_map[tid] = f'I-{chunk}'
        else:
            outer_map[tid] = f'B-{chunk}'; prev_outer = chunk

    clause_label = build_clause_spans(tokens)
    inner_map, clause_map = {}, {}
    prev_inner = prev_clause = None
    for tid, form, upos, deprel, head_id in tokens:
        cl = clause_label[tid]
        if cl is None:
            inner_map[tid] = 'O'; clause_map[tid] = 'O'
            prev_inner = prev_clause = None
        else:
            ic = outer_chunk(upos)
            if ic == 'O':
                inner_map[tid] = 'O'; prev_inner = None
            elif ic == prev_inner:
                inner_map[tid] = f'I-{ic}'
            else:
                inner_map[tid] = f'B-{ic}'; prev_inner = ic
            if cl == prev_clause:
                clause_map[tid] = f'I-{cl}'
            else:
                clause_map[tid] = f'B-{cl}'; prev_clause = cl

    return [(form, outer_map[tid], inner_map[tid], clause_map[tid])
            for tid, form, upos, deprel, head_id in tokens]

def save_nested_bio(all_sentences, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for sentence in all_sentences:
            for form, outer, inner, clause in sentence:
                f.write(f'{form}\t{outer}\t{inner}\t{clause}\n')
            f.write('\n')
    print(f'Kaydedildi → {output_path}')

def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    for split, fname in [('train','tr_boun-ud-train.conllu'),
                         ('dev',  'tr_boun-ud-dev.conllu'),
                         ('test', 'tr_boun-ud-test.conllu')]:
        print(f'{split} okunuyor...')
        sentences = read_conllu(os.path.join(base_path, fname))
        print(f'  {len(sentences)} cümle')
        nested = [convert_sentence(s) for s in sentences]
        save_nested_bio(nested, os.path.join(base_path, f'{split}_bio_nested.txt'))
    print('\nTamamlandı!')

if __name__ == '__main__':
    main()