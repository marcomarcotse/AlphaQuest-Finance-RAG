from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class CompanySpec:
    name: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class MetricSpec:
    metric_id: str
    label: str
    row_aliases: tuple[str, ...]
    query_terms: tuple[str, ...]
    note: str = ""


@dataclass(frozen=True)
class FinancialFact:
    company: str
    year: str
    metric_id: str
    metric_label: str
    value: int
    source_file: str
    filing_year: str
    page: int
    row_label: str
    confidence: int
    column_index: int
    note: str = ""


@dataclass(frozen=True)
class QueryIntent:
    question: str
    metric_id: str | None
    year: str | None
    companies: tuple[str, ...]
    order: str
    is_comparative: bool


@dataclass(frozen=True)
class QueryAnswer:
    used_facts: bool
    markdown: str
    intent: QueryIntent
    facts: tuple[FinancialFact, ...] = ()
    missing_companies: tuple[str, ...] = ()
    reason: str = ""


COMPANIES: tuple[CompanySpec, ...] = (
    CompanySpec("Apple", ("apple", "aapl")),
    CompanySpec("Amazon", ("amazon", "amzn")),
    CompanySpec("Alphabet", ("alphabet", "google", "goog", "googl")),
    CompanySpec("Meta", ("meta", "facebook", "fb")),
    CompanySpec("Microsoft", ("microsoft", "msft")),
    CompanySpec("Nvidia", ("nvidia", "nvda")),
    CompanySpec("Tesla", ("tesla", "tsla")),
)


METRICS: dict[str, MetricSpec] = {
    "revenue": MetricSpec(
        metric_id="revenue",
        label="Revenue",
        row_aliases=(
            "total net sales",
            "total revenue",
            "total revenues",
            "net sales",
            "revenues",
            "revenue",
        ),
        query_terms=(
            "revenue",
            "revenues",
            "sales",
            "net sales",
            "turnover",
            "營收",
            "收入",
        ),
    ),
    "net_income": MetricSpec(
        metric_id="net_income",
        label="Net income",
        row_aliases=("net income", "net income (loss)", "net earnings"),
        query_terms=(
            "net income",
            "net earnings",
            "profit",
            "profits",
            "earnings",
            "net profit",
            "淨利",
            "淨收入",
            "盈利",
        ),
    ),
    "operating_income": MetricSpec(
        metric_id="operating_income",
        label="Operating income",
        row_aliases=("operating income", "income from operations"),
        query_terms=(
            "operating income",
            "income from operations",
            "operating profit",
            "營業收入",
            "營運收入",
        ),
    ),
    "research_and_development": MetricSpec(
        metric_id="research_and_development",
        label="R&D / technology expense",
        row_aliases=(
            "research and development",
            "research and development expenses",
            "technology and infrastructure",
        ),
        query_terms=(
            "r&d",
            "research and development",
            "research expense",
            "development expense",
            "technology and infrastructure",
            "研發",
            "研究開發",
        ),
        note=(
            "Amazon reports Technology and infrastructure rather than a pure "
            "R&D line item, so that row is used as the closest comparable proxy."
        ),
    ),
}


COMPARATIVE_TERMS = (
    "highest",
    "lowest",
    "biggest",
    "smallest",
    "largest",
    "least",
    "most",
    "best",
    "worst",
    "compare",
    "comparison",
    "rank",
    "ranking",
    "top",
    "bottom",
    "list all",
    "everyone",
    "all companies",
    "magnificent",
    "最高",
    "最低",
    "最大",
    "最小",
    "比較",
    "排名",
    "全部",
)

ASCENDING_TERMS = ("lowest", "smallest", "least", "worst", "bottom", "最低", "最小")
DESCENDING_TERMS = ("highest", "largest", "biggest", "most", "best", "top", "最高", "最大")

BUNDLED_FACT_STORE_PATH = Path("app_data") / "financial_facts.json"
FACT_STORE_PATH = Path("storage") / "financial_facts.json"
_NUMBER_RE = re.compile(r"\(?\s*-?\s*\d{1,3}(?:,\d{3})+\s*\)?|\(?\s*-?\s*\d{4,6}\s*\)?")
_ASCII_WORD_RE = re.compile(r"\b{}\b")


def normalize_label(value: str) -> str:
    value = value.replace("\xa0", " ").replace("—", "-")
    value = value.strip().lower()
    value = re.sub(r"[$%]", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.rstrip(":").strip()


def get_file_metadata(file_path: str | Path) -> dict[str, str]:
    filename = Path(file_path).name.lower()
    metadata: dict[str, str] = {"company": "Unknown", "year": "Unknown"}

    for spec in COMPANIES:
        if any(alias in filename for alias in spec.aliases):
            metadata["company"] = spec.name
            break

    year_match = re.search(r"20\d{2}", filename)
    if year_match:
        metadata["year"] = year_match.group(0)

    return metadata


def parse_money_values(line: str) -> list[int]:
    if "%" in line:
        return []

    values: list[int] = []
    for match in _NUMBER_RE.finditer(line.replace("\xa0", " ")):
        token = match.group(0).strip()
        negative = "(" in token or token.startswith("-")
        clean = re.sub(r"[^\d]", "", token)
        if not clean:
            continue

        value = int(clean)
        if 1900 <= value <= 2099:
            continue
        if value < 10:
            continue

        values.append(-value if negative else value)

    return values


def _line_has_text(line: str) -> bool:
    return bool(re.search(r"[A-Za-z]", line))


def _collect_row_values(lines: list[str], start_index: int, expected_count: int) -> list[int]:
    values: list[int] = []

    for index in range(start_index + 1, min(len(lines), start_index + 35)):
        line = lines[index].strip()
        if not line:
            continue

        normalized = normalize_label(line)
        if normalized in {"$", "$ $"}:
            continue

        parsed = parse_money_values(line)
        if not values and not parsed and _line_has_text(line):
            # This is a category heading such as "Revenue:" followed by
            # "Product" or "Automotive sales", not a numeric fact row.
            return []

        if values and not parsed and _line_has_text(line):
            break

        values.extend(parsed)
        if len(values) >= expected_count:
            return values[:expected_count]

    return values[:expected_count]


def _statement_page_score(text: str) -> int:
    lower = text.lower()
    score = 0

    normalized_lines = {normalize_label(line) for line in text.splitlines()}
    if normalized_lines.intersection(
        {
            "consolidated statements of operations",
            "consolidated statements of income",
            "income statements",
        }
    ):
        score += 15

    if any(
        phrase in lower
        for phrase in (
            "consolidated statements of operations",
            "consolidated statements of income",
            "income statements",
        )
    ):
        score += 60
    if "in millions" in lower:
        score += 15
    if "year ended" in lower or "years ended" in lower:
        score += 10

    for spec in METRICS.values():
        for alias in spec.row_aliases:
            if alias in lower:
                score += 2

    if "expressed as a percentage of revenue" in lower:
        score -= 45
    if "the following table sets forth" in lower:
        score -= 10
    if "reportable segment" in lower:
        score -= 20
    if "non-gaap" in lower:
        score -= 15

    return score


def _extract_period_years(lines: list[str], company: str, filing_year: str) -> list[str]:
    normalized_lines = [normalize_label(line) for line in lines]
    starts = [
        index
        for index, line in enumerate(normalized_lines[:80])
        if "year ended" in line or "years ended" in line
    ]

    years: list[str] = []
    if starts:
        for line in lines[starts[0] : min(len(lines), starts[0] + 25)]:
            for year in re.findall(r"20\d{2}", line):
                if year not in years:
                    years.append(year)
            if len(years) >= 3:
                break

    if len(years) < 2:
        for line in lines[:60]:
            for year in re.findall(r"20\d{2}", line):
                if year not in years:
                    years.append(year)
            if len(years) >= 3:
                break

    if company == "Nvidia" and filing_year != "Unknown" and years:
        # Nvidia's fiscal year ends in late January. The local files are named by
        # the calendar year that the fiscal period mostly covers, so map the
        # first statement column back to the filename year for cross-company use.
        first_year = int(years[0])
        filing_year_int = int(filing_year)
        if first_year > filing_year_int:
            return [str(filing_year_int - offset) for offset in range(len(years))]

    return years[:3]


def _row_priority(metric: MetricSpec, row_label: str) -> int:
    normalized = normalize_label(row_label)
    try:
        index = metric.row_aliases.index(normalized)
    except ValueError:
        return 0
    return (len(metric.row_aliases) - index) * 5


def extract_facts_from_pdf(pdf_path: str | Path) -> list[FinancialFact]:
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("PyMuPDF is required to extract financial facts.") from exc

    path = Path(pdf_path)
    metadata = get_file_metadata(path)
    company = metadata["company"]
    filing_year = metadata["year"]
    facts: list[FinancialFact] = []

    if company == "Unknown" or filing_year == "Unknown":
        return facts

    document = fitz.open(path)
    try:
        for page_index, page in enumerate(document):
            text = page.get_text()
            page_score = _statement_page_score(text)
            if page_score < 20:
                continue

            lines = [line.strip() for line in text.splitlines() if line.strip()]
            years = _extract_period_years(lines, company, filing_year)
            if len(years) < 2:
                continue

            normalized_lines = [normalize_label(line) for line in lines]
            for metric in METRICS.values():
                for line_index, normalized in enumerate(normalized_lines):
                    if normalized not in metric.row_aliases:
                        continue

                    values = _collect_row_values(lines, line_index, len(years))
                    if len(values) < len(years):
                        continue

                    row_label = lines[line_index]
                    confidence = page_score + _row_priority(metric, row_label)
                    note = ""
                    if (
                        metric.metric_id == "research_and_development"
                        and normalize_label(row_label) == "technology and infrastructure"
                    ):
                        note = METRICS["research_and_development"].note

                    for column_index, (year, value) in enumerate(zip(years, values)):
                        facts.append(
                            FinancialFact(
                                company=company,
                                year=year,
                                metric_id=metric.metric_id,
                                metric_label=metric.label,
                                value=value,
                                source_file=path.name,
                                filing_year=filing_year,
                                page=page_index + 1,
                                row_label=row_label,
                                confidence=confidence,
                                column_index=column_index,
                                note=note,
                            )
                        )
                    break
    finally:
        document.close()

    return facts


def _fact_selection_score(fact: FinancialFact) -> int:
    score = fact.confidence
    if fact.filing_year == fact.year:
        score += 25
    if fact.column_index == 0:
        score += 5
    return score


def dedupe_facts(facts: Iterable[FinancialFact]) -> list[FinancialFact]:
    best: dict[tuple[str, str, str], FinancialFact] = {}

    for fact in facts:
        key = (fact.company, fact.year, fact.metric_id)
        current = best.get(key)
        if current is None or _fact_selection_score(fact) > _fact_selection_score(current):
            best[key] = fact

    return sorted(best.values(), key=lambda item: (item.metric_id, item.year, item.company))


def build_fact_store(
    data_dir: str | Path = "data",
    output_path: str | Path | None = FACT_STORE_PATH,
) -> list[FinancialFact]:
    data_path = Path(data_dir)
    if not data_path.exists():
        return []

    extracted: list[FinancialFact] = []
    for pdf_path in sorted(data_path.glob("*.pdf")):
        extracted.extend(extract_facts_from_pdf(pdf_path))

    facts = dedupe_facts(extracted)
    if output_path is not None:
        save_facts(facts, output_path)
    return facts


def save_facts(facts: Iterable[FinancialFact], output_path: str | Path = FACT_STORE_PATH) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "facts": [asdict(fact) for fact in facts],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_facts(path: str | Path = FACT_STORE_PATH) -> list[FinancialFact]:
    fact_path = Path(path)
    if not fact_path.exists():
        return []

    payload = json.loads(fact_path.read_text(encoding="utf-8"))
    return [FinancialFact(**item) for item in payload.get("facts", [])]


def load_or_build_facts(
    data_dir: str | Path = "data",
    fact_path: str | Path = FACT_STORE_PATH,
) -> list[FinancialFact]:
    facts = load_facts(fact_path)
    if facts:
        return facts
    bundled_facts = load_facts(BUNDLED_FACT_STORE_PATH)
    if bundled_facts:
        return bundled_facts
    return build_fact_store(data_dir=data_dir, output_path=fact_path)


def available_companies(facts: Iterable[FinancialFact]) -> list[str]:
    return sorted({fact.company for fact in facts})


def available_years(facts: Iterable[FinancialFact]) -> list[str]:
    return sorted({fact.year for fact in facts})


def _contains_term(text: str, term: str) -> bool:
    if re.search(r"[A-Za-z0-9]", term):
        return bool(re.search(_ASCII_WORD_RE.pattern.format(re.escape(term)), text))
    return term in text


def detect_metric(question: str) -> str | None:
    lower = question.lower()
    matches: list[tuple[int, str]] = []
    for metric_id, spec in METRICS.items():
        for term in spec.query_terms:
            if term.lower() in lower:
                matches.append((len(term), metric_id))

    if not matches:
        return None

    return sorted(matches, reverse=True)[0][1]


def detect_companies(question: str) -> tuple[str, ...]:
    lower = question.lower()
    found: list[str] = []
    for spec in COMPANIES:
        for alias in spec.aliases:
            if _contains_term(lower, alias):
                found.append(spec.name)
                break
    return tuple(found)


def detect_intent(question: str, facts: Iterable[FinancialFact]) -> QueryIntent:
    fact_list = list(facts)
    lower = question.lower()
    year_matches = re.findall(r"20\d{2}", question)
    years = available_years(fact_list)
    year = year_matches[-1] if year_matches else (years[-1] if years else None)
    metric_id = detect_metric(question)
    companies = detect_companies(question)

    is_comparative = any(term in lower for term in COMPARATIVE_TERMS)
    if not companies and (is_comparative or metric_id):
        companies = tuple(available_companies(fact_list))

    order = "desc"
    if any(term in lower for term in ASCENDING_TERMS):
        order = "asc"
    elif any(term in lower for term in DESCENDING_TERMS):
        order = "desc"

    return QueryIntent(
        question=question,
        metric_id=metric_id,
        year=year,
        companies=companies,
        order=order,
        is_comparative=is_comparative or len(companies) > 1,
    )


def _format_money(value: int) -> str:
    if value < 0:
        return f"-${abs(value):,} million"
    return f"${value:,} million"


def find_facts(
    facts: Iterable[FinancialFact],
    metric_id: str,
    year: str,
    companies: Iterable[str],
) -> tuple[list[FinancialFact], list[str]]:
    by_key = {(fact.company, fact.year, fact.metric_id): fact for fact in facts}
    selected: list[FinancialFact] = []
    missing: list[str] = []

    for company in companies:
        fact = by_key.get((company, year, metric_id))
        if fact is None:
            missing.append(company)
        else:
            selected.append(fact)

    return selected, missing


def answer_financial_question(question: str, facts: Iterable[FinancialFact]) -> QueryAnswer:
    fact_list = list(facts)
    intent = detect_intent(question, fact_list)
    if not intent.metric_id:
        return QueryAnswer(
            used_facts=False,
            markdown="",
            intent=intent,
            reason="No supported financial metric was detected.",
        )

    if not intent.year:
        return QueryAnswer(
            used_facts=False,
            markdown="",
            intent=intent,
            reason="No fiscal year is available in the fact store.",
        )

    companies = intent.companies or tuple(available_companies(fact_list))
    selected, missing = find_facts(fact_list, intent.metric_id, intent.year, companies)
    if not selected:
        markdown = (
            f"I could not find {METRICS[intent.metric_id].label} facts for "
            f"{intent.year} in the local fact store."
        )
        return QueryAnswer(
            used_facts=True,
            markdown=markdown,
            intent=intent,
            missing_companies=tuple(missing),
            reason="No matching facts were found.",
        )

    reverse = intent.order == "desc"
    ranked = tuple(sorted(selected, key=lambda fact: fact.value, reverse=reverse))
    metric = METRICS[intent.metric_id]
    leader = ranked[0]

    if len(ranked) == 1:
        intro = (
            f"**{leader.company} reported {metric.label} of "
            f"{_format_money(leader.value)} in {intent.year}.**"
        )
    elif intent.order == "asc":
        intro = (
            f"**{leader.company} had the lowest {metric.label} in {intent.year}: "
            f"{_format_money(leader.value)}.**"
        )
    else:
        intro = (
            f"**{leader.company} had the highest {metric.label} in {intent.year}: "
            f"{_format_money(leader.value)}.**"
        )

    lines = [
        "### Analysis Result",
        intro,
        "",
        "| Rank | Company | Value | Source |",
        "|---:|---|---:|---|",
    ]
    for rank, fact in enumerate(ranked, 1):
        source = f"{fact.source_file}, p. {fact.page} ({fact.row_label})"
        lines.append(f"| {rank} | {fact.company} | {_format_money(fact.value)} | {source} |")

    notes = sorted({fact.note for fact in ranked if fact.note})
    if missing:
        notes.append("Missing companies: " + ", ".join(missing))
    if notes:
        lines.extend(["", "#### Notes"])
        lines.extend(f"- {note}" for note in notes)

    return QueryAnswer(
        used_facts=True,
        markdown="\n".join(lines),
        intent=intent,
        facts=ranked,
        missing_companies=tuple(missing),
    )
