"""
Settings and Path Configurations
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    bolt_home: Path = Path.home() / ".bolt"

    @property
    def db_path(self) -> Path:
        return self.bolt_home / "state.db"

    class Config:
        env_prefix = "BOLT_"

settings = Settings()
