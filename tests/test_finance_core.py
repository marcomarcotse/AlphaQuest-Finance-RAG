from __future__ import annotations

import unittest
from pathlib import Path

from finance_core import (
    answer_financial_question,
    build_fact_store,
    detect_companies,
    detect_metric,
    parse_money_values,
)


class FinanceCoreUnitTests(unittest.TestCase):
    def test_parse_money_values_handles_parentheses_and_years(self) -> None:
        self.assertEqual(parse_money_values("$ (2,722) 2024 30,425"), [-2722, 30425])

    def test_detect_metric_and_companies(self) -> None:
        self.assertEqual(detect_metric("Compare Apple and Microsoft revenue in 2024"), "revenue")
        self.assertEqual(detect_companies("Compare Apple and Microsoft revenue"), ("Apple", "Microsoft"))


@unittest.skipUnless(Path("data").exists(), "local PDF data is not available")
class FinanceCorePdfIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.facts = build_fact_store(output_path=None)
        cls.by_key = {
            (fact.company, fact.year, fact.metric_id): fact
            for fact in cls.facts
        }

    def test_extracts_total_revenue_rows_not_category_rows(self) -> None:
        self.assertEqual(self.by_key[("Microsoft", "2024", "revenue")].value, 245_122)
        self.assertEqual(self.by_key[("Tesla", "2024", "revenue")].value, 97_690)

    def test_extracts_amazon_net_income_loss_row(self) -> None:
        self.assertEqual(self.by_key[("Amazon", "2024", "net_income")].value, 59_248)
        self.assertEqual(self.by_key[("Amazon", "2023", "net_income")].value, 30_425)

    def test_maps_nvidia_fiscal_year_to_project_year(self) -> None:
        self.assertEqual(self.by_key[("Nvidia", "2024", "revenue")].value, 130_497)
        self.assertEqual(self.by_key[("Nvidia", "2023", "revenue")].value, 60_922)

    def test_answers_ranked_questions_deterministically(self) -> None:
        revenue_answer = answer_financial_question(
            "Which company had the highest revenue in 2024?",
            self.facts,
        )
        self.assertTrue(revenue_answer.used_facts)
        self.assertEqual(revenue_answer.facts[0].company, "Amazon")
        self.assertEqual(revenue_answer.facts[0].value, 637_959)

        income_answer = answer_financial_question(
            "Which company had the highest net income in 2023?",
            self.facts,
        )
        self.assertEqual(income_answer.facts[0].company, "Apple")
        self.assertEqual(income_answer.facts[0].value, 96_995)


if __name__ == "__main__":
    unittest.main()
