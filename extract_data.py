import re

import pandas as pd
import pdfplumber

MONTHS_ID = {
    "januari": 1, "februari": 2, "maret": 3, "april": 4, "mei": 5, "juni": 6,
    "juli": 7, "agustus": 8, "september": 9, "oktober": 10, "november": 11, "desember": 12,
}

AQMS_COLS = ["pm10", "pm25", "so2", "co", "o3", "no2", "hc"]
ISPU_COLS = ["ispu_pm10", "ispu_pm25", "ispu_so2", "ispu_co", "ispu_o3", "ispu_no2", "ispu_hc"]


def parse_id_number(value: str):
    if value is None:
        return None
    value = value.strip()
    if value in ("", "-"):
        return None
    value = value.replace(".", "").replace(",", ".")
    try:
        return float(value)
    except ValueError:
        return None


def parse_date_id(text: str):
    """Parse '1 Januari 2024' -> pandas.Timestamp; return None for summary rows."""
    m = re.match(r"(\d{1,2})\s+(\w+)\s+(\d{4})", text.strip())
    if not m:
        return None
    day, month_name, year = m.groups()
    month = MONTHS_ID.get(month_name.lower())
    if month is None:
        return None
    try:
        return pd.Timestamp(year=int(year), month=month, day=int(day))
    except ValueError:
        return None


def extract_daily_table(pdf_path: str, value_cols: list[str]) -> pd.DataFrame:
    records = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                for row in table:
                    if not row or len(row) < 2:
                        continue
                    date = parse_date_id(str(row[1]) if row[1] else "")
                    if date is None:
                        continue 

                    raw_values = row[2 : 2 + len(value_cols)]
                    parsed = [parse_id_number(v) for v in raw_values]
                    while len(parsed) < len(value_cols):
                        parsed.append(None)

                    record = {"date": date}
                    record.update(dict(zip(value_cols, parsed)))
                    records.append(record)

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    return df


def main():
    aqms = extract_daily_table(
        "data_source/Data Konsentrasi Rata - Rata Harian AQMS Tahun 2024.pdf",
        AQMS_COLS,
    )
    ispu = extract_daily_table(
        "data_source/ISPU Kota Yogyakarta Tahun 2024.pdf",
        ISPU_COLS,
    )

    print(f"AQMS rows parsed: {len(aqms)} | ISPU rows parsed: {len(ispu)}")

    merged = pd.merge(aqms, ispu, on="date", how="outer").sort_values("date").reset_index(drop=True)


    full_range = pd.DataFrame({"date": pd.date_range("2024-01-01", "2024-12-31", freq="D")})
    merged = pd.merge(full_range, merged, on="date", how="left")

    out_path = "data_source/yogyakarta_air_quality_2024.csv"
    merged.to_csv(out_path, index=False)
    print(f"Saved {out_path} with shape {merged.shape}")
    print(merged.head(10))
    print("\nMissing values per column:")
    print(merged.isna().sum())


if __name__ == "__main__":
    main()
