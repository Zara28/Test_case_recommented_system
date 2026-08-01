import re

import pandas as pd

from VectorizerSevice import Vectorizer


class Process:

    def __init__(self):
        self.data = self.load_and_prepare_catalog("catalog_excel.csv")
        self.vectorizer = self.init_vectorizer(self.data)

    def normalize_text(self, text: str) -> str:
        """
        Очищает текст от мусора, приводит к нижнему регистру.
        Сохраняем цифры, точки и запятые, так как они критичны для размеров.
        """
        if not isinstance(text, str):
            return ""
        text = text.lower()
        # Заменяем всё, кроме букв, цифр, пробелов, точек и запятых, на пробелы
        text = re.sub(r'[^\w\s\.,]', ' ', text)
        # Убираем лишние пробелы
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def load_and_prepare_catalog(self, csv_path: str) -> pd.DataFrame:
        """
        Загружает каталог и подготавливает поле для поиска.
        """
        # Склеиваем название и единицы измерения.
        # Часто покупатель ищет именно по связке (например, "саморезы 75 пачка")
        self.data['search_text'] = self.data['name'].fillna('') + " " + self.data['unit'].fillna('')
        self.data['search_text'] = self.data['search_text'].apply(self.normalize_text)

        return self.data

    def init_vectorizer(self, data):

        df_catalog = pd.DataFrame(data)
        df_catalog['search_text'] = df_catalog['name'] + " " + df_catalog['unit']
        df_catalog['search_text'] = df_catalog['search_text'].apply(self.normalize_text)

        vector = Vectorizer(df_catalog['search_text'])

        return vector

    def get_answer(self, messages):
        results = []
        messages_normal = [self.normalize_text(msg) for msg in messages]

        candidates_indexes = self.vectorizer.search_candidates(messages_normal)
        for i, indexes in enumerate(candidates_indexes):
            candidates = []
            for idx in indexes['candidates']:
                candidates.append({
                    "sku": self.data.iloc[idx['index']]['sku'],
                    "confidence": idx['confidence']
                })

            results.append({
                "message": messages[i],
                "status": indexes["status"],
                "candidates": candidates
            })

        return results
