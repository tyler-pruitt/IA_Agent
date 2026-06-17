"""Fetch market-wide RQData technical indicators with daily cache control.

This script intentionally fetches technical indicator factors only. It does not
request financial statement fields, valuation fundamentals, or WorldQuant alpha
factors. The traffic-control pattern mirrors `fetch_rqdatac_panel_data.py`:

- stock universe cache
- trading-date loop
- per-date parquet cache by default
- stock chunks and factor chunks
- resumable cache hits
- progress logging and summary file
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

import rqdatac as rq


DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2025-12-31"
DEFAULT_OUTPUT_DIR = "outputs/raw_rqdatac_technical_indicators"
DEFAULT_FACTOR_FILE = "ricequant_factor_names.txt"
DEFAULT_UNIVERSE_CSV = "outputs/raw_rqdatac_database/metadata_universe_2020q1_2025q4.csv"

PRICE_FIELDS = ["close", "volume", "total_turnover"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch market-wide RQData technical indicator factors only."
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="Start trading date, YYYY-MM-DD.")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE, help="End trading date, YYYY-MM-DD.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for caches and summary.")
    parser.add_argument("--factor-file", default=DEFAULT_FACTOR_FILE, help="Text file containing available RQData factor names.")
    parser.add_argument(
        "--universe-csv",
        default=DEFAULT_UNIVERSE_CSV,
        help="Optional local universe CSV. Falls back to rq.all_instruments if missing.",
    )
    parser.add_argument("--stock-chunk-size", type=int, default=500, help="Stock chunk size for rq.get_factor.")
    parser.add_argument("--factor-chunk-size", type=int, default=80, help="Factor chunk size for rq.get_factor.")
    parser.add_argument("--progress-every", type=int, default=25, help="Print progress every N trading dates.")
    parser.add_argument("--refresh-cache", action="store_true", help="Ignore existing per-date caches.")
    parser.add_argument(
        "--include-price",
        action="store_true",
        help="Also fetch close/volume/total_turnover via rq.get_price. Default is indicator factors only.",
    )
    parser.add_argument(
        "--write-combined",
        action="store_true",
        help="Concatenate per-date caches into one file. For long ranges this can be very large.",
    )
    parser.add_argument(
        "--output-format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Storage format for per-date caches and optional combined output. Default: parquet.",
    )
    parser.add_argument(
        "--include-factors",
        default="",
        help="Comma-separated extra factor names to include if present in the factor file.",
    )
    parser.add_argument(
        "--exclude-factors",
        default="",
        help="Comma-separated factor names to exclude from the selected technical factors.",
    )
    parser.add_argument(
        "--max-factors",
        type=int,
        default=0,
        help="Optional cap for debugging. 0 means no cap.",
    )
    return parser.parse_args()


def chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def load_factor_names(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Factor name file not found: {path}")
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [name for name in names if name and not name.startswith("#")]


def select_technical_factors(
    all_names: list[str],
    include_factors: str = "",
    exclude_factors: str = "",
    max_factors: int = 0,
) -> list[str]:
    """Select standard technical indicators and exclude WorldQuant factors."""
    available = set(all_names)
    selected: list[str] = []
    for name in all_names:
        if name.startswith("WorldQuant_alpha"):
            break
        if name and not name.startswith("WorldQuant_alpha"):
            selected.append(name)

    extras = [item.strip() for item in include_factors.split(",") if item.strip()]
    for factor in extras:
        if factor in available and not factor.startswith("WorldQuant_alpha") and factor not in selected:
            selected.append(factor)

    excludes = {item.strip() for item in exclude_factors.split(",") if item.strip()}
    selected = [factor for factor in selected if factor not in excludes and not factor.startswith("WorldQuant_alpha")]
    if max_factors > 0:
        selected = selected[:max_factors]
    return selected


def load_universe(rq_module, universe_csv: str) -> pd.DataFrame:
    path = Path(universe_csv)
    if path.exists():
        universe = pd.read_csv(path, low_memory=False)
        if "order_book_id" not in universe.columns:
            raise ValueError(f"Universe CSV is missing order_book_id: {path}")
        if "status" in universe.columns:
            universe = universe.loc[universe["status"].astype(str).str.lower().eq("active")]
        return universe.dropna(subset=["order_book_id"]).drop_duplicates("order_book_id")

    instruments = rq_module.all_instruments(type="CS", market="cn")
    universe = instruments.copy()
    if "order_book_id" not in universe.columns:
        raise ValueError("rq.all_instruments result is missing order_book_id")
    return universe.dropna(subset=["order_book_id"]).drop_duplicates("order_book_id")


def get_trading_dates(rq_module, start_date: str, end_date: str) -> list[pd.Timestamp]:
    try:
        dates = rq_module.get_trading_dates(start_date, end_date, market="cn")
    except TypeError:
        dates = rq_module.get_trading_dates(start_date, end_date)
    return [pd.Timestamp(date_value) for date_value in dates]


def fetch_date_indicators(
    rq_module,
    ids: list[str],
    factor_names: list[str],
    trading_date: pd.Timestamp,
    stock_chunk_size: int,
    factor_chunk_size: int,
    include_price: bool,
) -> pd.DataFrame:
    date_parts: list[pd.DataFrame] = []
    date_value = trading_date.date()

    for id_chunk in chunks(ids, stock_chunk_size):
        chunk_frame: pd.DataFrame | None = None
        for factor_chunk in chunks(factor_names, factor_chunk_size):
            factors = rq_module.get_factor(
                id_chunk,
                factor_chunk,
                start_date=date_value,
                end_date=date_value,
                expect_df=True,
            )
            factor_frame = factors.reset_index() if factors is not None and len(factors) else pd.DataFrame()
            if factor_frame.empty:
                continue
            if chunk_frame is None:
                chunk_frame = factor_frame
            else:
                chunk_frame = chunk_frame.merge(
                    factor_frame,
                    on=["order_book_id", "date"],
                    how="outer",
                )

        if include_price:
            prices = rq_module.get_price(
                id_chunk,
                start_date=date_value,
                end_date=date_value,
                frequency="1d",
                fields=PRICE_FIELDS,
                adjust_type="none",
                skip_suspended=True,
                expect_df=True,
            )
            price_frame = prices.reset_index() if prices is not None and len(prices) else pd.DataFrame()
            if not price_frame.empty:
                if chunk_frame is None or chunk_frame.empty:
                    chunk_frame = price_frame
                else:
                    chunk_frame = chunk_frame.merge(
                        price_frame,
                        on=["order_book_id", "date"],
                        how="left",
                    )

        if chunk_frame is not None and not chunk_frame.empty:
            date_parts.append(chunk_frame)

    if not date_parts:
        return pd.DataFrame({"order_book_id": ids, "date": date_value})

    result = pd.concat(date_parts, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    return result.sort_values(["date", "order_book_id"]).reset_index(drop=True)


def should_print_progress(date_number: int, total_dates: int, progress_every: int, id_count: int) -> bool:
    if date_number in {1, total_dates}:
        return True
    if progress_every > 0 and date_number % progress_every == 0:
        return True
    return id_count >= 5000


def cache_path_for_date(cache_dir: Path, trading_date: pd.Timestamp, output_format: str) -> Path:
    suffix = "parquet" if output_format == "parquet" else "csv"
    return cache_dir / f"technical_{trading_date:%Y%m%d}.{suffix}"


def read_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
        return
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def frame_columns(path: Path) -> set[str]:
    if path.suffix == ".parquet":
        return set(pd.read_parquet(path, engine="pyarrow").columns)
    return set(pd.read_csv(path, nrows=0).columns)


def cache_has_expected_columns(cache_path: Path, factor_names: list[str], include_price: bool) -> bool:
    if not cache_path.exists():
        return False
    try:
        columns = frame_columns(cache_path)
    except Exception:
        return False
    expected = {"order_book_id", "date", *factor_names}
    if include_price:
        expected.update(PRICE_FIELDS)
    return expected.issubset(columns)


def write_summary(
    output_dir: Path,
    cache_dir: Path,
    universe_count: int,
    trading_dates: list[pd.Timestamp],
    factor_names: list[str],
    cache_hits: int,
    cache_misses: int,
    output_format: str,
    combined_path: Path | None,
) -> None:
    lines = ["RQData Technical Indicator Fetch Summary", ""]
    lines.append(f"Date range: {trading_dates[0].date()} -> {trading_dates[-1].date()}" if trading_dates else "Date range: empty")
    lines.append(f"Trading dates: {len(trading_dates)}")
    lines.append(f"Companies: {universe_count}")
    lines.append(f"Technical factors: {len(factor_names)}")
    lines.append(f"WorldQuant included: false")
    lines.append(f"Financial fields included: false")
    lines.append(f"Cache hits: {cache_hits}")
    lines.append(f"Cache misses: {cache_misses}")
    lines.append(f"Cache directory: {cache_dir}")
    lines.append(f"Output format: {output_format}")
    if combined_path is not None:
        lines.append(f"Combined file: {combined_path}")
    lines.append("")
    lines.append("Factors:")
    lines.extend(f"- {factor}" for factor in factor_names)
    (output_dir / "technical_indicator_fetch_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rq.init()

    output_dir = Path(args.output_dir)
    cache_dir = output_dir / f"technical_fetch_cache_{args.start_date.replace('-', '')}_{args.end_date.replace('-', '')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_factor_names = load_factor_names(Path(args.factor_file))
    factor_names = select_technical_factors(
        all_factor_names,
        include_factors=args.include_factors,
        exclude_factors=args.exclude_factors,
        max_factors=args.max_factors,
    )
    if not factor_names:
        raise RuntimeError("No technical indicator factors selected.")
    print(f"technical_factors_selected={len(factor_names)} worldquant_included=false financial_fields=false")

    universe = load_universe(rq, args.universe_csv)
    ids = universe["order_book_id"].astype(str).tolist()
    trading_dates = get_trading_dates(rq, args.start_date, args.end_date)
    print(f"universe_ids={len(ids)} trading_dates={len(trading_dates)}")

    cache_hits = 0
    cache_misses = 0
    combined_parts: list[pd.DataFrame] = []

    for date_number, trading_date in enumerate(trading_dates, start=1):
        cache_path = cache_path_for_date(cache_dir, trading_date, args.output_format)
        cache_valid = (
            cache_path.exists()
            and not args.refresh_cache
            and cache_has_expected_columns(cache_path, factor_names, args.include_price)
        )
        if cache_valid:
            cache_hits += 1
            if should_print_progress(date_number, len(trading_dates), args.progress_every, len(ids)):
                print(f"technical_date={date_number}/{len(trading_dates)} date={trading_date.date()} ids={len(ids)} cache=hit")
            if args.write_combined:
                combined_parts.append(read_frame(cache_path))
            continue

        cache_misses += 1
        if should_print_progress(date_number, len(trading_dates), args.progress_every, len(ids)):
            print(f"technical_date={date_number}/{len(trading_dates)} date={trading_date.date()} ids={len(ids)} cache=miss")
        date_frame = fetch_date_indicators(
            rq,
            ids,
            factor_names,
            trading_date,
            args.stock_chunk_size,
            args.factor_chunk_size,
            args.include_price,
        )
        write_frame(cache_path, date_frame)
        if args.write_combined:
            combined_parts.append(date_frame)

    combined_path = None
    if args.write_combined:
        combined_suffix = "parquet" if args.output_format == "parquet" else "csv"
        combined_path = output_dir / f"technical_indicators_{args.start_date.replace('-', '')}_{args.end_date.replace('-', '')}.{combined_suffix}"
        combined = pd.concat(combined_parts, ignore_index=True) if combined_parts else pd.DataFrame()
        write_frame(combined_path, combined)
        print(f"combined_path={combined_path} rows={len(combined)}")

    write_summary(
        output_dir,
        cache_dir,
        len(ids),
        trading_dates,
        factor_names,
        cache_hits,
        cache_misses,
        args.output_format,
        combined_path,
    )
    print(f"technical_cache_dir={cache_dir}")
    print(f"cache_hits={cache_hits} cache_misses={cache_misses}")


if __name__ == "__main__":
    main()
