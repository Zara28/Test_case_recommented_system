import json
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


def match_messages(messages: list, df: pd.DataFrame, vectorizer: TfidfVectorizer, tfidf_matrix):
    results = []

    for msg in messages:
        norm_query = normalize_text(msg)
        if not norm_query:
            results.append({"message": msg, "status": "not_found", "candidates": []})
            continue

        query_vec = vectorizer.transform([norm_query])
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-3:][::-1]

        candidates = []
        for idx in top_indices:
            score = round(float(similarities[idx]), 3)
            if score >= T_LOW:
                candidates.append({
                    "sku": df.iloc[idx]['sku'],
                    "name": df.iloc[idx]['name'],
                    "price": df.iloc[idx]['price'],
                    "confidence": score
                })

        # --- БЛОК ЛОГИКИ СТАТУСОВ ---
        if not candidates:
            status = "not_found"
        elif candidates[0]['confidence'] >= T_HIGH:
            status = "matched"
            candidates = [candidates[0]]  # Оставляем только топ-1 кандидат
        else:
            status = "ambiguous"

        results.append({
            "message": msg,
            "status": status,
            "candidates": candidates
        })

    return {"results": results}


def print_human_readable(results_dict):
    """Выводит результаты так, как бы ответил AI-агент в чате."""
    print("\n" + "=" * 50)
    print("ЭМУЛЯЦИЯ ОТВЕТОВ AI-АГЕНТА")
    print("=" * 50)

    for item in results_dict["results"]:
        print(f"\nПокупатель: «{item['message']}»")
        print(f"Статус системы: [{item['status'].upper()}]")

        if item['status'] == "not_found":
            print("Агент: Простите, я не понял запрос или такого товара нет в нашем каталоге.")

        elif item['status'] == "matched":
            c = item['candidates'][0]
            print(
                f"Агент: Отличный выбор! Добавляю в корзину: {c['name']} (SKU: {c['sku']}, Уверенность: {c['confidence']})")

        elif item['status'] == "ambiguous":
            print("Агент: Я нашел несколько похожих вариантов. Уточните, какой именно вам нужен:")
            for i, c in enumerate(item['candidates'], 1):
                print(f"  {i}. {c['name']} (SKU: {c['sku']}, Уверенность: {c['confidence']})")

data = load_and_prepare_catalog("catalog_excel.csv")
df_catalog = pd.DataFrame(data)
df_catalog['search_text'] = df_catalog['name'] + " " + df_catalog['unit']
df_catalog['search_text'] = df_catalog['search_text'].apply(normalize_text)

vector, matrix = train_vectorizer(df_catalog['search_text'])

T_HIGH = 0.85  # Порог уверенного ответа
T_LOW = 0.30

with open("messages.txt") as file:
    lines = [line.rstrip() for line in file]
    final_output = match_messages(lines, df_catalog, vector, matrix)
    json_output = {"results": []}
    for r in final_output["results"]:
        clean_candidates = [{"sku": c["sku"], "confidence": c["confidence"]} for c in r["candidates"]]
        json_output["results"].append({
            "message": r["message"],
            "status": r["status"],
            "candidates": clean_candidates
        })

    # 5. Выводим красивый лог
    print_human_readable(final_output)