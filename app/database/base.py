"""Repository interface. Services depend on this, never on a specific database."""
from abc import ABC, abstractmethod
from typing import Optional


class Repository(ABC):
    # ---- results (raw and processed are separate tables) ----
    @abstractmethod
    def init(self) -> None: ...
    @abstractmethod
    def add_results(self, items: list) -> list:
        """items: [{'raw': dict, 'number': int, 'period_id': str|None, 'timestamp': str|None}]. Atomic. Returns processed rows."""
    @abstractmethod
    def get_numbers(self) -> list: ...
    @abstractmethod
    def get_meta(self) -> list:
        """[{'seq','period_id','timestamp'}] ascending (for data-quality checks)."""
    @abstractmethod
    def list_results(self, limit: int, offset: int) -> tuple: ...   # (rows newest first, total)
    @abstractmethod
    def last_result(self) -> Optional[dict]: ...
    @abstractmethod
    def has_period(self, period_id: str) -> bool: ...
    # ---- predictions / outcomes ----
    @abstractmethod
    def save_prediction(self, pred: dict) -> int: ...
    @abstractmethod
    def get_prediction_by_target(self, target_seq: int) -> Optional[dict]: ...
    @abstractmethod
    def expire_pending_except(self, target_seq: int) -> None: ...
    @abstractmethod
    def resolve_prediction(self, prediction_id: int, outcome: dict) -> None: ...
    @abstractmethod
    def list_predictions(self, limit: int, offset: int) -> tuple: ...   # (rows newest first, total)
    @abstractmethod
    def all_outcomes(self) -> list: ...                                 # ascending by target_seq
    # ---- backtests / performance snapshots ----
    @abstractmethod
    def save_backtest(self, summary: dict, rows: list) -> int: ...
    @abstractmethod
    def latest_backtest(self, with_rows: bool = False) -> Optional[dict]: ...
    @abstractmethod
    def save_snapshot(self, snap: dict) -> None: ...
    @abstractmethod
    def list_snapshots(self, limit: int = 500) -> list: ...
