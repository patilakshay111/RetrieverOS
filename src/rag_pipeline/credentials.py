# credentials.py: encrypts each user's OpenAI key before storing it in Postgres

from datetime import datetime, timezone

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import DateTime, LargeBinary, String, create_engine, delete, text
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class Base(DeclarativeBase):
    pass


class UserOpenAIKey(Base):
    __tablename__ = "user_openai_keys"

    user_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    encrypted_key: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class UserCredentialStore:
    def __init__(self, database_url: str, encryption_key: str):
        try:
            self._cipher = Fernet(encryption_key.encode("ascii"))
        except (ValueError, UnicodeEncodeError) as error:
            raise ValueError(
                "USER_KEY_ENCRYPTION_KEY must be a valid Fernet key."
            ) from error
        engine_options = {"pool_pre_ping": True}
        if database_url.startswith("sqlite"):
            engine_options.update(
                poolclass=StaticPool,
                connect_args={"check_same_thread": False},
            )
        else:
            engine_options.update(pool_size=5, max_overflow=10)
        self._engine = create_engine(database_url, **engine_options)
        if database_url.startswith("sqlite"):
            Base.metadata.create_all(self._engine)

    def has_key(self, user_id: str, workspace_id: str) -> bool:
        with Session(self._engine) as session:
            return (
                session.get(UserOpenAIKey, (user_id, workspace_id)) is not None
            )

    def get_key(self, user_id: str, workspace_id: str) -> str | None:
        with Session(self._engine) as session:
            record = session.get(UserOpenAIKey, (user_id, workspace_id))
            if record is None:
                return None
            try:
                return self._cipher.decrypt(record.encrypted_key).decode("utf-8")
            except (InvalidToken, UnicodeDecodeError) as error:
                raise RuntimeError(
                    "Stored OpenAI credential could not be decrypted."
                ) from error

    def set_key(self, user_id: str, workspace_id: str, api_key: str) -> None:
        encrypted = self._cipher.encrypt(api_key.encode("utf-8"))
        with Session(self._engine) as session, session.begin():
            record = session.get(UserOpenAIKey, (user_id, workspace_id))
            if record is None:
                record = UserOpenAIKey(
                    user_id=user_id,
                    workspace_id=workspace_id,
                    encrypted_key=encrypted,
                    updated_at=datetime.now(timezone.utc),
                )
                session.add(record)
            else:
                record.encrypted_key = encrypted
                record.updated_at = datetime.now(timezone.utc)

    def delete_key(self, user_id: str, workspace_id: str) -> None:
        with Session(self._engine) as session, session.begin():
            session.execute(
                delete(UserOpenAIKey).where(
                    UserOpenAIKey.user_id == user_id,
                    UserOpenAIKey.workspace_id == workspace_id,
                )
            )

    def check(self) -> None:
        with self._engine.connect() as connection:
            connection.execute(text("SELECT 1"))
