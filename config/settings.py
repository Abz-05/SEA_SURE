"""
Application settings and environment configuration.

This module centralizes all environment variables and configuration
using Pydantic for validation and type safety.
"""

import os
from typing import Optional
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database Configuration
    db_host: str = Field(default="localhost", description="Database host")
    db_port: int = Field(default=5432, description="Database port")
    db_name: str = Field(default="seasure_db", description="Database name")
    db_user: Optional[str] = Field(default=None, description="Database user")
    db_password: Optional[str] = Field(default=None, description="Database password")
    db_pool_min: int = Field(default=1, description="Minimum database connections")
    db_pool_max: int = Field(default=20, description="Maximum database connections")
    
    # Security
    secret_key: str = Field(
        default="default-secret-key-change-me",
        description="Application secret key"
    )
    qr_secret_key: str = Field(
        default="default-qr-secret-key-change-me",
        description="QR code signing secret"
    )
    use_qr_signature: bool = Field(
        default=True,
        description="Enable QR code digital signatures"
    )
    
    # External APIs
    openweathermap_api_key: Optional[str] = Field(
        default=None,
        description="OpenWeatherMap API key"
    )
    twilio_account_sid: Optional[str] = Field(
        default=None,
        description="Twilio account SID"
    )
    twilio_auth_token: Optional[str] = Field(
        default=None,
        description="Twilio auth token"
    )
    twilio_from_number: Optional[str] = Field(
        default=None,
        description="Twilio phone number"
    )
    
    # Storage Paths
    qr_storage_path: Path = Field(
        default=Path("storage/qr"),
        description="QR code storage directory"
    )
    fish_images_path: Path = Field(
        default=Path("storage/fish_images"),
        description="Fish images storage directory"
    )
    uploads_path: Path = Field(
        default=Path("uploads"),
        description="Uploads directory"
    )
    offline_cache_path: Path = Field(
        default=Path("storage/offline_cache"),
        description="Offline cache directory"
    )
    logs_path: Path = Field(
        default=Path("logs"),
        description="Logs directory"
    )
    
    # Application Settings
    dev_mode: bool = Field(
        default=False,
        description="Enable development mode"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    max_upload_size_mb: int = Field(
        default=10,
        description="Maximum upload size in MB"
    )
    session_timeout_minutes: int = Field(
        default=60,
        description="Session timeout in minutes"
    )
    
    # Rate Limiting
    login_rate_limit: int = Field(
        default=5,
        description="Max login attempts per time window"
    )
    login_rate_window_seconds: int = Field(
        default=60,
        description="Login rate limit time window"
    )
    api_rate_limit: int = Field(
        default=100,
        description="Max API calls per time window"
    )
    api_rate_window_seconds: int = Field(
        default=60,
        description="API rate limit time window"
    )
    
    # ML Model Settings
    ml_model_path: Optional[Path] = Field(
        default=None,
        description="Path to ML model files"
    )
    ml_confidence_threshold: float = Field(
        default=0.7,
        description="Minimum confidence for ML predictions"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_prefix="",
        extra="ignore"
    )
    
    @validator("qr_storage_path", "fish_images_path", "uploads_path", 
               "offline_cache_path", "logs_path", pre=True)
    def create_directories(cls, v):
        """Ensure storage directories exist."""
        path = Path(v) if not isinstance(v, Path) else v
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v_upper
    
    @property
    def database_url(self) -> str:
        """Get database connection URL."""
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.dev_mode
    
    @property
    def twilio_enabled(self) -> bool:
        """Check if Twilio is configured."""
        return all([
            self.twilio_account_sid,
            self.twilio_auth_token,
            self.twilio_from_number
        ])
    
    @property
    def weather_api_enabled(self) -> bool:
        """Check if weather API is configured."""
        return self.openweathermap_api_key is not None


# Global settings instance
settings = Settings()


# Convenience function for getting settings
def get_settings() -> Settings:
    """Get application settings instance."""
    return settings
