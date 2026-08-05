"""Categorical encoding that survives categories it has never seen.

Eight cities appear only in the validation set and account for 12% of the loads
we must price. A plain lookup would raise or emit NaN on all of them, so every
encoder here maps an unfamiliar value to a reserved code and flags the row.
The model still has coordinates to place the load, which is the real fallback.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from src.config import Config
from src.logger import get_logger

logger = get_logger(__name__)

# Columns encoded by name. Lane-level encoding is deliberately excluded:
# lanes carry a median of 12 loads each, far too thin to be reliable.
CATEGORICAL_COLUMNS = ["pickup", "delivery", "equipment"]


class EncodingError(Exception):
    """Raised when a categorical column cannot be encoded."""


@dataclass(frozen=True)
class OrdinalEncoder:
    """Maps category names to integers, with a reserved code for unknowns.

    Args:
        mapping: Category name to integer code, per column.
        unknown_value: Code returned for anything not seen during training.
    """

    mapping: dict[str, dict[str, int]]
    unknown_value: int = -1
    columns: list[str] = field(default_factory=lambda: list(CATEGORICAL_COLUMNS))

    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        columns: list[str] | None = None,
        unknown_value: int = -1,
    ) -> OrdinalEncoder:
        """Learn the category codes from training data.

        Pickup and delivery share one set of codes so that a city keeps the
        same identity whichever end of the lane it appears on.

        Args:
            df: Training frame.
            columns: Columns to encode. Defaults to pickup, delivery, equipment.
            unknown_value: Code to use for unseen categories.

        Returns:
            The fitted encoder.

        Raises:
            EncodingError: If a requested column is not in the frame.
        """
        columns = columns or list(CATEGORICAL_COLUMNS)

        if missing := [c for c in columns if c not in df.columns]:
            raise EncodingError(f"cannot encode, columns not found: {missing}")

        mapping: dict[str, dict[str, int]] = {}

        # One shared vocabulary for cities, so Atlanta means the same thing as
        # an origin and as a destination.
        city_columns = [c for c in columns if c in ("pickup", "delivery")]
        if city_columns:
            cities = sorted(set(pd.concat([df[c] for c in city_columns]).dropna()))
            city_codes = {str(city): code for code, city in enumerate(cities)}
            for column in city_columns:
                mapping[column] = city_codes

        for column in columns:
            if column not in mapping:
                values = sorted(df[column].dropna().unique())
                mapping[column] = {str(v): code for code, v in enumerate(values)}

        encoder = cls(mapping=mapping, unknown_value=unknown_value, columns=columns)
        logger.info(
            "Fitted encoder: %s",
            {col: len(codes) for col, codes in encoder.mapping.items()},
        )
        return encoder

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode the categorical columns and flag unknown values.

        Args:
            df: Frame to encode.

        Returns:
            The frame with a `<column>_code` column per input, plus
            `has_unknown_city` marking rows the model has no history for.

        Raises:
            EncodingError: If the encoder has not been fitted.
        """
        if not self.mapping:
            raise EncodingError("encoder has not been fitted")

        out = df.copy()
        unknown_flags = []

        for column in self.columns:
            if column not in out.columns:
                raise EncodingError(f"'{column}' not found — cannot encode")

            codes = out[column].map(self.mapping[column])
            unseen = codes.isna()

            out[f"{column}_code"] = codes.fillna(self.unknown_value).astype("int32")

            if column in ("pickup", "delivery"):
                unknown_flags.append(unseen)

            if unseen.any():
                names = sorted(set(out.loc[unseen, column]))
                logger.debug(
                    "%s: %s rows in %s categories not seen in training %s",
                    column,
                    f"{int(unseen.sum()):,}",
                    len(names),
                    names[:8],
                )

        # One flag the model can split on, rather than several it cannot.
        if unknown_flags:
            combined = unknown_flags[0]
            for flag in unknown_flags[1:]:
                combined = combined | flag
            out["has_unknown_city"] = combined.astype("int8")

            if combined.any():
                logger.info(
                    "%s rows (%.1f%%) involve a city not seen in training",
                    f"{int(combined.sum()):,}",
                    combined.mean() * 100,
                )

        return out

    def known_categories(self, column: str) -> set[str]:
        """List the categories learned for a column.

        Args:
            column: Column name.

        Returns:
            Category names seen during training.
        """
        return set(self.mapping.get(column, {}))

    def to_json(self, path: Path) -> None:
        """Save the encoder next to the model.

        Args:
            path: Destination JSON file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        logger.debug("Saved encoder to %s", path)

    @classmethod
    def from_json(cls, path: Path) -> OrdinalEncoder:
        """Load an encoder saved by a previous training run.

        Args:
            path: JSON file written by to_json.

        Returns:
            The stored encoder.

        Raises:
            EncodingError: If the file is missing or malformed.
        """
        if not path.is_file():
            raise EncodingError(f"encoder not found: {path}")

        try:
            return cls(**json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, TypeError) as exc:
            raise EncodingError(f"could not read {path}: {exc}") from exc


def encoded_feature_names(config: Config) -> list[str]:
    """List the encoded column names, in build order.

    Args:
        config: Loaded project configuration.

    Returns:
        Code columns followed by the unknown-city flag.
    """
    del config  # Signature kept consistent with the other feature modules.
    return [f"{column}_code" for column in CATEGORICAL_COLUMNS] + ["has_unknown_city"]