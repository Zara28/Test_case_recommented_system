import re

import pandas as pd

from VectorizerSevice import Vectorizer
from config import Settings


class Process:

    def __init__(self, settings: Settings):
        self.data = self.load_and_prepare_catalog(settings.csv_path)
        self.vectorizer = self.init_vectorizer(self.data, settings)

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
        data = pd.read_csv(csv_path)

        data['search_text'] = data['name'].fillna('') + " " + data['unit'].fillna('')
        data['search_text'] = data['search_text'].apply(self.normalize_text)

        return data

    def init_vectorizer(self, data, settings):
        """
        Инициализирует набор векторов для поиска по каталогу
        """

        df_catalog = pd.DataFrame(data)
        df_catalog['search_text'] = df_catalog['name'] + " " + df_catalog['unit']
        df_catalog['search_text'] = df_catalog['search_text'].apply(self.normalize_text)

        vector = Vectorizer(df_catalog['search_text'], settings.t_high, settings.t_low)

        return vector

    def get_answer(self, messages):
        """
        Метод поиска ответов на заданные вопросы
        """
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
