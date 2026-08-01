import os
import pytest
import pandas as pd
from fastapi.testclient import TestClient

# === Подготовка тестовой среды ===
TEST_CSV_PATH = "test_catalog_mock.csv"
os.environ["CSV_PATH"] = TEST_CSV_PATH
os.environ["T_HIGH"] = "0.85"
os.environ["T_LOW"] = "0.30"


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """
    Фикстура, которая создает временный CSV-файл для тестов
    и подменяет переменные окружения ДО загрузки приложения.
    """

    data = {
        "sku": ["TEST-001", "TEST-002", "TEST-003", "TEST-004"],
        "name": ["Саморезы по дереву", "Саморезы по металлу", "Гвозди", "Кабель ВВГнг"],
        "unit": ["4.2х75 пачка", "4.2х75", "100 мм", "3х1.5"],
        "price": [100, 120, 50, 200]
    }
    pd.DataFrame(data).to_csv(TEST_CSV_PATH, index=False)

    os.environ["CSV_PATH"] = TEST_CSV_PATH
    os.environ["T_HIGH"] = "0.85"
    os.environ["T_LOW"] = "0.30"

    yield

    if os.path.exists(TEST_CSV_PATH):
        os.remove(TEST_CSV_PATH)


from main import app


@pytest.fixture()
def client():
    """
    Используем контекстный менеджер TestClient, чтобы корректно
    отработал lifespan (загрузка данных при старте).
    """
    with TestClient(app) as c:
        yield c


def test_match_exact_product(client):
    """Позитивный сценарий: точное совпадение (matched)"""
    payload = {"messages": ["Привет! ищу саморезы по дереву 4.2х75 пачка"]}
    response = client.post("/match", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert len(data["results"]) == 1
    result = data["results"][0]

    assert result["status"] == "matched"
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["sku"] == "TEST-001"
    assert result["candidates"][0]["confidence"] >= 0.85


def test_match_ambiguous_product(client):
    """Пограничный сценарий: неоднозначный запрос (ambiguous)"""
    payload = {"messages": ["саморезы 4.2"]}
    response = client.post("/match", json=payload)

    assert response.status_code == 200
    result = response.json()["results"][0]

    assert result["status"] == "ambiguous"
    # Должно вернуться несколько вариантов
    assert len(result["candidates"]) > 1


def test_match_not_found(client):
    """Негативный сценарий: товара нет в базе (not_found)"""
    payload = {"messages": ["Ищу смартфон", "qwerty"]}
    response = client.post("/match", json=payload)

    assert response.status_code == 200
    results = response.json()["results"]

    assert len(results) == 2
    assert results[0]["status"] == "not_found"
    assert len(results[0]["candidates"]) == 0
    assert results[1]["status"] == "not_found"
    assert len(results[1]["candidates"]) == 0


def test_match_empty_string(client):
    """Негативный сценарий: пустая строка"""
    payload = {"messages": [""]}
    response = client.post("/match", json=payload)

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["status"] == "not_found"


def test_invalid_request_format(client):
    """Негативный сценарий: неправильный формат JSON (должна отработать валидация Pydantic)"""
    payload = {"texts": ["кабель"]}
    response = client.post("/match", json=payload)

    # Ожидаем статус 422 Unprocessable Entity
    assert response.status_code == 422