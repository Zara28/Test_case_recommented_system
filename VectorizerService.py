import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Vectorizer:
    def __init__(self, corpus: pd.Series, t_high: float, t_low: float):
        self.vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

        self.T_HIGH = t_high
        self.T_LOW = t_low

    def match_status(self, candidates):
        """
        Матчинг статусов
        """
        if not candidates:
            status = "not_found"
        elif candidates[0]['confidence'] >= self.T_HIGH:
            status = "matched"
            candidates = [candidates[0]]
        else:
            status = "ambiguous"

        return status, candidates

    def search_candidates(self, messages: list):
        """
        Реализация векторного поиска
        """
        results = []

        for msg in messages:
            if not msg:
                results.append({
                    "status": "not_found",
                    "candidates": []
                })
                continue

            query_vec = self.vectorizer.transform([msg])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            top_indices = similarities.argsort()[-3:][::-1]

            candidates = []
            for idx in top_indices:
                score = round(float(similarities[idx]), 3)
                if score >= self.T_LOW:
                    candidates.append({
                        "index": idx,
                        "confidence": score
                    })

            status, candidates = self.match_status(candidates)

            results.append({
                "status": status,
                "candidates": candidates
            })

        return results
