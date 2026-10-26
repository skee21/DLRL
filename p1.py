!pip install -q gensim
from gensim.models import Word2Vec

corpus = [
    "The king and queen ruled the kingdom.",
    "A man and a woman are walking in the rain.",
    "Teachers share knowledge with their students."
]

tokenized = [s.lower().split() for s in corpus]
model = Word2Vec(tokenized, vector_size=50, window=3, min_count=1, sg=1, epochs=100)

for w in model.wv.index_to_key:
    print(f"{w}: {model.wv[w][0]:.4f}")