from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Инициализация настроек
    """
    csv_path: str = "catalog_excel.csv"
    t_high: float = 0.85
    t_low: float = 0.30

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
