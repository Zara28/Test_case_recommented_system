import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def normalize_text(text: str) -> str:
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


def load_and_prepare_catalog(csv_path: str) -> pd.DataFrame:
    """
    Загружает каталог и подготавливает поле для поиска.
    """
    df = pd.read_csv(csv_path)

    # Склеиваем название и единицы измерения.
    # Часто покупатель ищет именно по связке (например, "саморезы 75 пачка")
    df['search_text'] = df['name'].fillna('') + " " + df['unit'].fillna('')
    df['search_text'] = df['search_text'].apply(normalize_text)

    return df


def train_vectorizer(corpus: pd.Series):
    """
    Обучает TF-IDF на символьных N-граммах.
    analyzer='char_wb' создает N-граммы внутри границ слов (полезно против опечаток).
    """
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix


def match_product(query: str, df: pd.DataFrame, vectorizer: TfidfVectorizer, tfidf_matrix, top_k: int = 3):
    """
    Векторизует запрос и возвращает топ-K кандидатов с их уверенностью (confidence).
    """
    norm_query = normalize_text(query)

    # Если после нормализации запрос пустой
    if not norm_query:
        return []

    query_vec = vectorizer.transform([norm_query])

    # Считаем косинусное расстояние между запросом и всеми векторами каталога
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Получаем индексы топ-K результатов (сортировка по возрастанию, поэтому берем с конца)
    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        score = similarities[idx]
        if score > 0.05:  # Отсекаем совсем нулевые/случайные совпадения
            results.append({
                "sku": df.iloc[idx]['sku'],
                "name": df.iloc[idx]['name'],
                "confidence": round(float(score), 3)
            })
    return results

data = load_and_prepare_catalog("catalog_excel.csv")
df_catalog = pd.DataFrame(data)
df_catalog['search_text'] = df_catalog['name'] + " " + df_catalog['unit']
df_catalog['search_text'] = df_catalog['search_text'].apply(normalize_text)

vector, matrix = train_vectorizer(df_catalog['search_text'])

with open("messages.txt") as file:
    lines = [line.rstrip() for line in file]
    for line in lines:
        result = match_product(line, df_catalog, vector, matrix)
        print(f"Для запроса {line} найдены результаты: {result}")