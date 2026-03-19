from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pandas as pd

AGE_COLUMN_INDEX = {
    '18-34': 3,
    '35-54': 4,
    '55+': 5,
}

PARTY_NAME_MAP = {
    'Liberal Party': 'Liberal',
    'Conservative Party': 'Conservative',
    'New Democratic Party (NDP)': 'NDP',
    'Bloc Québécois': 'Bloc Québécois',
    'Green Party': 'Green',
    'People’s Party': 'Others',
    "People's Party": 'Others',
    'Another party / An independent candidate': 'Others',
}

EXPECTED_CORE_LABELS = {
    'Liberal Party',
    'Conservative Party',
    'New Democratic Party (NDP)',
}


def pdf_to_layout_lines(pdf_path: Path) -> list[str]:
    try:
        result = subprocess.run(
            ['pdftotext', '-layout', str(pdf_path), '-'],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError('`pdftotext` is required to parse the Research Co PDF age tables.') from exc
    return [line.replace('\x0c', '').rstrip() for line in result.stdout.splitlines()]


def extract_decided_voter_block(lines: list[str]) -> list[tuple[str, list[float]]]:
    starts = [i for i, line in enumerate(lines) if line.strip() == 'Decided Voters']
    candidates: list[list[tuple[str, list[float]]]] = []

    for start in starts:
        rows: list[tuple[str, list[float]]] = []
        for line in lines[start + 1:]:
            stripped = line.strip()
            if not stripped:
                if rows:
                    break
                continue
            if '%' not in stripped:
                if rows:
                    break
                continue
            match = re.match(r'^(.*?)\s{2,}(\d+%.*)$', stripped)
            if not match:
                if rows:
                    break
                continue
            label = match.group(1).strip()
            values = [float(x) for x in re.findall(r'(\d+)%', match.group(2))]
            rows.append((label, values))

        labels = {label for label, _ in rows}
        has_expected = EXPECTED_CORE_LABELS <= labels
        has_exclusion = any('Undecided' in label or 'Not sure' in label for label, _ in rows)
        if rows and has_expected and not has_exclusion:
            candidates.append(rows)

    if not candidates:
        raise RuntimeError('Could not find a decided-voter age table in the PDF text.')

    return candidates[-1]


def parse_researchco_decided_voter_age_table(pdf_path: Path, year: int) -> pd.DataFrame:
    rows = extract_decided_voter_block(pdf_to_layout_lines(pdf_path))
    parsed = []

    for raw_label, values in rows:
        mapped_party = PARTY_NAME_MAP.get(raw_label)
        if mapped_party is None:
            continue
        if len(values) < max(AGE_COLUMN_INDEX.values()) + 1:
            continue
        for age_band, idx in AGE_COLUMN_INDEX.items():
            parsed.append(
                {
                    'Year': year,
                    'AgeBand': age_band,
                    'Party': mapped_party,
                    'SupportPct': float(values[idx]),
                    'RawPartyLabel': raw_label,
                    'SourceFile': str(pdf_path),
                }
            )

    df = pd.DataFrame(parsed)
    if df.empty:
        raise RuntimeError(f'No age cross-tab values parsed from {pdf_path}')

    df = df.groupby(['Year', 'AgeBand', 'Party'], as_index=False).agg(
        SupportPct=('SupportPct', 'sum'),
        RawPartyLabel=('RawPartyLabel', lambda s: ' + '.join(sorted(set(s)))),
        SourceFile=('SourceFile', 'first'),
    )
    return df.sort_values(['Year', 'AgeBand', 'Party']).reset_index(drop=True)


def parse_multiple_age_tables(pdf_by_year: dict[int, Path]) -> pd.DataFrame:
    frames = [parse_researchco_decided_voter_age_table(path, year) for year, path in sorted(pdf_by_year.items())]
    return pd.concat(frames, ignore_index=True)
