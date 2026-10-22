"""Importable tokenization, fixed TF-IDF configurations and label-free scoring."""
import hashlib
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

SCHEMES = ('char_1', 'char_1_2', 'word_1')
_TOKENIZER = jieba.Tokenizer()  # isolated instance; shipped default dictionary


def jieba_tokens(text):
    return [t for t in _TOKENIZER.cut(text, cut_all=False, HMM=False) if not t.isspace()]


def dictionary_info():
    stream = _TOKENIZER.get_dict_file()
    try:
        return {'path': str(stream.name), 'sha256': hashlib.sha256(stream.read()).hexdigest(),
                'jieba_version': jieba.__version__, 'cut_all': False, 'HMM': False,
                'dictionary': 'jieba shipped default; no user dictionary or add_word/del_word',
                'filter': 'only pure-whitespace tokens'}
    finally:
        stream.close()


def make_vectorizer(scheme):
    if scheme not in SCHEMES:
        raise ValueError(scheme)
    return TfidfVectorizer(
        analyzer='word' if scheme == 'word_1' else 'char',
        ngram_range=(1, 2) if scheme == 'char_1_2' else (1, 1),
        tokenizer=jieba_tokens if scheme == 'word_1' else None,
        token_pattern=None if scheme == 'word_1' else r'(?u)\b\w\w+\b',
        use_idf=True, smooth_idf=True, norm='l2', min_df=1, max_df=1.0,
        max_features=None, sublinear_tf=False, binary=False, lowercase=False,
        stop_words=None, dtype=np.float64)


def row_cosine(left, right):
    """Only vectors, never labels. No N-by-N matrix or dense feature arrays."""
    if left.shape != right.shape:
        raise ValueError('paired vector shapes differ')
    a = np.asarray(left.getnnz(axis=1) == 0)
    b = np.asarray(right.getnnz(axis=1) == 0)
    scores = np.asarray(left.multiply(right).sum(axis=1)).reshape(-1)
    scores[a | b] = 0.0
    if not np.isfinite(scores).all():
        raise ValueError('nonfinite similarity')
    if ((scores < -1e-12) | (scores > 1 + 1e-12)).any():
        raise ValueError('similarity exceeds tolerance')
    outside = int(((scores < 0) | (scores > 1)).sum())
    return np.clip(scores, 0, 1), {
        'sentence1_zero': int(a.sum()), 'sentence2_zero': int(b.sum()),
        'either_zero': int((a | b).sum()), 'both_zero': int((a & b).sum()),
        'clipped_within_tolerance': outside}


def transform_pairs(vectorizer, sentence1, sentence2):
    return row_cosine(vectorizer.transform(sentence1), vectorizer.transform(sentence2))
