from abc import ABC, abstractmethod
import datetime
import logging
import sqlite3
import threading
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

class DBManager(ABC):

    @abstractmethod
    def add_history(
            self,
            memory_id: str,
            old_memory: Optional[str],
            new_memory: Optional[str],
            event: str,
            *,
            created_at: Optional[str] = None,
            updated_at: Optional[str] = None,
            is_deleted: int = 0,
            actor_id: Optional[str] = None,
            role: Optional[str] = None,
    ) -> None:
        pass

    @abstractmethod
    def get_history(self, memory_id: str) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


class SQLiteManager(DBManager):
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._migrate_history_table()
        self._create_history_table()

    def _migrate_history_table(self) -> None:
        """
        If a pre-existing history table had the old group-chat columns,
        rename it, create the new schema, copy the intersecting data, then
        drop the old table.
        """
        with self._lock:
            try:
                # Start a transaction
                self.connection.execute("BEGIN")
                cur = self.connection.cursor()

                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='history'")
                if cur.fetchone() is None:
                    self.connection.execute("COMMIT")
                    return  # nothing to migrate

                cur.execute("PRAGMA table_info(history)")
                old_cols = {row[1] for row in cur.fetchall()}

                expected_cols = {
                    "id",
                    "memory_id",
                    "old_memory",
                    "new_memory",
                    "event",
                    "created_at",
                    "updated_at",
                    "is_deleted",
                    "actor_id",
                    "role",
                }

                if old_cols == expected_cols:
                    self.connection.execute("COMMIT")
                    return

                logger.info("Migrating history table to new schema (no convo columns).")

                # Clean up any existing history_old table from previous failed migration
                cur.execute("DROP TABLE IF EXISTS history_old")

                # Rename the current history table
                cur.execute("ALTER TABLE history RENAME TO history_old")

                # Create the new history table with updated schema
                cur.execute(
                    """
                    CREATE TABLE history (
                        id           TEXT PRIMARY KEY,
                        memory_id    TEXT,
                        old_memory   TEXT,
                        new_memory   TEXT,
                        event        TEXT,
                        created_at   DATETIME,
                        updated_at   DATETIME,
                        is_deleted   INTEGER,
                        actor_id     TEXT,
                        role         TEXT
                    )
                """
                )

                # Copy data from old table to new table
                intersecting = list(expected_cols & old_cols)
                if intersecting:
                    cols_csv = ", ".join(intersecting)
                    cur.execute(f"INSERT INTO history ({cols_csv}) SELECT {cols_csv} FROM history_old")

                # Drop the old table
                cur.execute("DROP TABLE history_old")

                # Commit the transaction
                self.connection.execute("COMMIT")
                logger.info("History table migration completed successfully.")

            except Exception as e:
                # Rollback the transaction on any error
                self.connection.execute("ROLLBACK")
                logger.error(f"History table migration failed: {e}")
                raise

    def _create_history_table(self) -> None:
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS history (
                        id           TEXT PRIMARY KEY,
                        memory_id    TEXT,
                        old_memory   TEXT,
                        new_memory   TEXT,
                        event        TEXT,
                        created_at   DATETIME,
                        updated_at   DATETIME,
                        is_deleted   INTEGER,
                        actor_id     TEXT,
                        role         TEXT
                    )
                """
                )
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to create history table: {e}")
                raise

    def add_history(
        self,
        memory_id: str,
        old_memory: Optional[str],
        new_memory: Optional[str],
        event: str,
        *,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        is_deleted: int = 0,
        actor_id: Optional[str] = None,
        role: Optional[str] = None,
    ) -> None:
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute(
                    """
                    INSERT INTO history (
                        id, memory_id, old_memory, new_memory, event,
                        created_at, updated_at, is_deleted, actor_id, role
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        str(uuid.uuid4()),
                        memory_id,
                        old_memory,
                        new_memory,
                        event,
                        created_at,
                        updated_at,
                        is_deleted,
                        actor_id,
                        role,
                    ),
                )
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to add history record: {e}")
                raise

    def get_history(self, memory_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self.connection.execute(
                """
                SELECT id, memory_id, old_memory, new_memory, event,
                       created_at, updated_at, is_deleted, actor_id, role
                FROM history
                WHERE memory_id = ?
                ORDER BY created_at ASC, DATETIME(updated_at) ASC
            """,
                (memory_id,),
            )
            rows = cur.fetchall()

        return [
            {
                "id": r[0],
                "memory_id": r[1],
                "old_memory": r[2],
                "new_memory": r[3],
                "event": r[4],
                "created_at": r[5],
                "updated_at": r[6],
                "is_deleted": bool(r[7]),
                "actor_id": r[8],
                "role": r[9],
            }
            for r in rows
        ]

    def reset(self) -> None:
        """Drop and recreate the history table."""
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute("DROP TABLE IF EXISTS history")
                self.connection.execute("COMMIT")
                self._create_history_table()
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to reset history table: {e}")
                raise

    def close(self) -> None:
        if self.connection:
            self.connection.close()
            self.connection = None

    def __del__(self):
        self.close()



from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    insert,
    select,
)
from sqlalchemy.engine.url import make_url

class MySQLManager(DBManager):

    # db_url = "mysql+pymysql://your_username:your_password@your_connect_ip:3306/history_db?charset=utf8mb4"
    def __init__(self, db_url: str):
        self._lock = threading.Lock()
        url_obj = make_url(db_url)
        db_name = url_obj.database
        if not db_name:
            raise ValueError("No database name specified in db_url for MySQL")

        try:
            self._engine = create_engine(db_url)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to database: {e}")

        self._metadata = MetaData()

        self._history_table_name = "history"

        self._history_table = Table(
            self._history_table_name,
            self._metadata,
            Column("id", String(36), primary_key=True),
            Column("memory_id", String(255), index=True),
            Column("old_memory", String),
            Column("new_memory", String),
            Column("event", String(255)),
            Column("created_at", DateTime),
            Column("updated_at", DateTime, index=True),
            Column("is_deleted", Integer, default=0),
            Column("actor_id", String),
            Column("role", String),
        )

        # create table
        self._create_history_table()


    def _create_history_table(self) -> None:
        """Create history table if it doesn't exist."""
        self._metadata.create_all(self._engine, tables=[self._history_table])


    def add_history(self, memory_id: str, old_memory: Optional[str], new_memory: Optional[str], event: str, *,
                    created_at: Optional[str] = None, updated_at: Optional[str] = None, is_deleted: int = 0,
                    actor_id: Optional[str] = None, role: Optional[str] = None) -> None:
        """
                Add a history record to the database.
                Returns:
                    The ID of the newly created history record
                """
        now = datetime.datetime.now()
        if created_at is None:
            created_at = now
        if updated_at is None:
            updated_at = now
        created_at = self._ensure_datetime(created_at)
        updated_at = self._ensure_datetime(updated_at)
        record_id = str(uuid.uuid4())
        with self._lock, self._engine.begin() as conn:
            stmt = insert(self._history_table).values(
                id=record_id,
                memory_id=memory_id,
                old_memory=old_memory,
                new_memory=new_memory,
                new_value=new_memory,
                event=event,
                created_at=created_at,
                updated_at=updated_at,
                is_deleted=is_deleted,
            )
            conn.execute(stmt)
        return record_id

    def get_history(self, memory_id: str) -> List[Dict[str, Any]]:
        """
                Get history records for a specific memory ID.
                Returns:
                    List of history records as dictionaries
                """
        with self._lock, self._engine.connect() as conn:
            query = (
                select(
                    self._history_table.c.id,
                    self._history_table.c.memory_id,
                    self._history_table.c.old_memory,
                    self._history_table.c.new_memory,
                    self._history_table.c.event,
                    self._history_table.c.created_at,
                    self._history_table.c.updated_at,
                    self._history_table.c.is_deleted,
                    self._history_table.c.actor_id,
                    self._history_table.c.role,
                )
                .where(
                    self._history_table.c.memory_id == memory_id,
                    self._history_table.c.is_deleted == 0,
                )
                .order_by(self._history_table.c.updated_at.asc())
            )
            result = conn.execute(query)
            rows = result.fetchall()
            return [
                {
                    "id": r[0],
                    "memory_id": r[1],
                    "old_memory": r[2],
                    "new_memory": r[3],
                    "event": r[4],
                    "created_at": r[5],
                    "updated_at": r[6],
                    "is_deleted": bool(r[7]),
                    "actor_id": r[8],
                    "role": r[9],
                }
                for r in rows
            ]

    def reset(self) -> None:
        """Reset database by dropping and recreating the history table."""
        with self._engine.connect() as conn:
            conn.execute(f"DELETE FROM {self._history_table_name}")
            conn.commit()
        pass

    def close(self) -> None:
        """Close database connections properly."""
        if self._engine:
            self._engine.dispose()
            self._engine = None

