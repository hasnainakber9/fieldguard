"""Skip-logic and constraint expressions: XLSForm and Python syntax, missing-value semantics and the safety whitelist."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fieldguard.expressions import Expression, ExpressionError, translate_xlsform
from fieldguard.io.columns import ColumnResolver


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "has_business": ["yes", "no", None, "yes"],
            "business_income": ["1000", None, None, "-99"],
            "age": ["34", "17", None, "70"],
            "group/crops": ["maize beans", "rice", None, "maize"],
            "n_children": ["2", "0", "1", "5"],
            "hh_size": ["4", "2", "3", "3"],
        }
    )


def evaluate(expr: str, df: pd.DataFrame | None = None) -> list[bool]:
    d = df if df is not None else frame()
    return Expression(expr).evaluate(d, ColumnResolver(d.columns)).tolist()


def test_xlsform_syntax_is_translated():
    assert translate_xlsform("${age} = 'x'") == "`age` == 'x'"
    assert translate_xlsform("${a} div 2 = 1") == "`a` / 2 == 1"
    assert "==" in translate_xlsform("a = 1")


def test_string_equality_treats_missing_as_empty():
    assert evaluate("has_business == 'yes'") == [True, False, False, True]
    assert evaluate("has_business != 'yes'") == [False, True, True, False]


def test_numeric_comparison_is_false_when_missing():
    assert evaluate("age >= 18") == [True, False, False, True]
    assert evaluate("n_children <= hh_size") == [True, True, True, False]


def test_xlsform_references_and_backticks_for_group_paths():
    assert evaluate("selected(`group/crops`, 'maize')") == [True, False, False, True]
    assert evaluate("${has_business} = 'yes' and ${age} >= 18") == [True, False, False, True]


def test_empty_filled_and_membership():
    assert evaluate("empty(business_income)") == [False, True, True, False]
    assert evaluate("filled(has_business)") == [True, True, False, True]
    assert evaluate("has_business in ['yes', 'maybe']") == [True, False, False, True]
    assert evaluate("has_business not in ['yes']") == [False, True, True, False]


def test_arithmetic_and_count_selected():
    assert evaluate("n_children + 1 == hh_size") == [False, False, False, False]
    assert evaluate("count_selected(`group/crops`) == 2") == [True, False, False, False]


def test_column_names_listed_without_functions():
    expr = Expression("selected(`group/crops`, 'maize') and age >= 18 and has_business == 'yes'")
    assert expr.column_names() == ["group/crops", "age", "has_business"]


@pytest.mark.parametrize(
    "bad",
    ["__import__('os')", "age.__class__", "lambda: 1", "[x for x in age]", "open('f')", "age if age else 1"],
)
def test_unsafe_syntax_is_refused(bad):
    with pytest.raises(ExpressionError):
        Expression(bad)


def test_unknown_column_fails_at_evaluation():
    with pytest.raises(ExpressionError, match="unknown column"):
        Expression("no_such_column == 1").evaluate(frame(), ColumnResolver(frame().columns))


def test_syntax_error_is_reported():
    with pytest.raises(ExpressionError, match="cannot parse"):
        Expression("age >=")


def test_result_is_boolean_series_aligned_to_frame():
    df = frame()
    out = Expression("age > 30").evaluate(df, ColumnResolver(df.columns))
    assert out.dtype == bool
    assert out.index.equals(df.index)
    assert np.array_equal(out.to_numpy(), np.array([True, False, False, True]))
