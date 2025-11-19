from pathlib import Path

import pandas as pd


def main() -> None:
    excel_path = Path(__file__).resolve().with_name("open_question_data.xlsx")

    if not excel_path.exists():
        raise FileNotFoundError(f"Could not find Excel file at {excel_path}")

    df = pd.read_excel(excel_path, nrows=0)

    print("Columns found in the Excel file:")
    for idx, column in enumerate(df.columns, start=1):
        print(f"{idx}. {column}")


if __name__ == "__main__":
    main()

