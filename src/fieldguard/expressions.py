"""Safe, vectorised evaluation of skip-logic / constraint expressions.

Expressions are parsed with :mod:`ast` and evaluated against a DataFrame using a strict whitelist
(no ``eval``, no attribute access, no arbitrary calls). Both Python-style and XLSForm-style syntax work::

    q5 == 'no'                      ${q5} = 'no'
    n_children <= hh_size           ${age} >= 18 and ${consent} = 'yes'
    selected(`grp/crops`, 'maize')  empty(q6)        q2 in ['a', 'b']

Column names that are not valid identifiers (``group/q1``) can be written in backticks or ``${...}``.
Missing values: string comparisons treat missing as "" (ODK semantics); numeric comparisons with a missing
operand are False.
"""

from __future__ import annotations

import ast
import operator
import re
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from fieldguard.io.columns import ColumnResolver


class ExpressionError(ValueError):
    pass


_XLS_REF = re.compile(r"\$\{([^}]+)\}")
_BACKTICK = re.compile(r"`([^`]+)`")
_SINGLE_EQ = re.compile(r"(?<![<>!=])=(?!=)")


def translate_xlsform(expr: str) -> str:
    """Translate XLSForm syntax to the Python subset (``${x}`` -> backticks, ``=`` -> ``==``)."""
    out = _XLS_REF.sub(lambda m: f"`{m.group(1).strip()}`", expr)
    # protect string literals before replacing '='
    parts = re.split(r"('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")", out)
    for i in range(0, len(parts), 2):
        seg = _SINGLE_EQ.sub("==", parts[i])
        seg = re.sub(r"\bdiv\b", "/", seg)
        seg = re.sub(r"\bmod\b", "%", seg)
        parts[i] = seg
    return "".join(parts)


def _norm_str(s: pd.Series) -> pd.Series:
    t = s.astype("str").where(s.notna(), "")
    return t.str.strip()


def _to_num(s: Any) -> Any:
    if isinstance(s, pd.Series):
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            return s.astype("float64")
        return pd.to_numeric(s.astype("str").where(s.notna()), errors="coerce").astype("float64")
    if isinstance(s, bool):
        return float(s)
    if isinstance(s, (int, float, np.number)):
        return float(s)
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan


def _is_numeric_like(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float, np.number)):
        return True
    if isinstance(value, pd.Series):
        if pd.api.types.is_bool_dtype(value):
            return False
        if pd.api.types.is_numeric_dtype(value):
            return True
        nonnull = value.dropna()
        if nonnull.empty:
            return False
        return bool(pd.to_numeric(nonnull.astype("str"), errors="coerce").notna().all())
    return False


_CMP: dict[type, Callable[[Any, Any], Any]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
_BIN: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
}
_LITERALS = {"true": True, "false": False, "True": True, "False": False, "None": None, "null": None}


class Expression:
    """A compiled expression; call :meth:`evaluate` with a DataFrame."""

    def __init__(self, source: str):
        self.source = source
        text = translate_xlsform(source.strip())
        self._names: dict[str, str] = {}

        def repl(m: re.Match[str]) -> str:
            key = f"__c{len(self._names)}"
            self._names[key] = m.group(1)
            return key

        text = _BACKTICK.sub(repl, text)
        try:
            self.tree = ast.parse(text, mode="eval")
        except SyntaxError as exc:
            raise ExpressionError(f"cannot parse expression {source!r}: {exc.msg}") from exc
        self._validate(self.tree)

    def column_names(self) -> list[str]:
        names: list[str] = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Name) and node.id not in _LITERALS and node.id not in _FUNCS:
                names.append(self._names.get(node.id, node.id))
        return list(dict.fromkeys(names))

    def _validate(self, tree: ast.AST) -> None:
        allowed = (
            ast.Expression,
            ast.BoolOp,
            ast.And,
            ast.Or,
            ast.UnaryOp,
            ast.Not,
            ast.USub,
            ast.UAdd,
            ast.Compare,
            ast.BinOp,
            ast.Name,
            ast.Load,
            ast.Constant,
            ast.List,
            ast.Tuple,
            ast.Call,
            ast.In,
            ast.NotIn,
            *_CMP.keys(),
            *_BIN.keys(),
        )
        for node in ast.walk(tree):
            if not isinstance(node, allowed):
                raise ExpressionError(f"unsupported syntax {type(node).__name__} in {self.source!r}")
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
                    raise ExpressionError(f"only functions {sorted(_FUNCS)} are allowed in {self.source!r}")
                if node.keywords:
                    raise ExpressionError("keyword arguments are not allowed")

    def evaluate(self, df: pd.DataFrame, resolver: ColumnResolver | None = None) -> pd.Series:
        resolver = resolver or ColumnResolver(df.columns)
        result = _Evaluator(df, resolver, self._names).visit(self.tree.body)
        if isinstance(result, pd.Series):
            if not pd.api.types.is_bool_dtype(result):
                result = result.fillna(False).astype(bool) if result.dtype == object else result.astype(bool)
            return result.reindex(df.index, fill_value=False).astype(bool)
        return pd.Series(bool(result), index=df.index)


def _f_empty(x: Any) -> Any:
    if isinstance(x, pd.Series):
        return x.isna() | (_norm_str(x) == "")
    return x is None or str(x).strip() == ""


def _f_filled(x: Any) -> Any:
    r = _f_empty(x)
    return ~r if isinstance(r, pd.Series) else not r


def _f_selected(x: Any, value: Any) -> Any:
    v = str(value).strip()
    if isinstance(x, pd.Series):
        return _norm_str(x).str.split().map(lambda toks: v in toks)
    return v in str(x).split()


def _f_number(x: Any) -> Any:
    return _to_num(x)


def _f_string(x: Any) -> Any:
    return _norm_str(x) if isinstance(x, pd.Series) else str(x)


def _f_count_selected(x: Any) -> Any:
    if isinstance(x, pd.Series):
        return _norm_str(x).str.split().map(len).astype("float64")
    return float(len(str(x).split()))


_FUNCS: dict[str, Callable[..., Any]] = {
    "empty": _f_empty,
    "filled": _f_filled,
    "notempty": _f_filled,
    "selected": _f_selected,
    "number": _f_number,
    "string": _f_string,
    "count_selected": _f_count_selected,
}


class _Evaluator:
    def __init__(self, df: pd.DataFrame, resolver: ColumnResolver, names: dict[str, str]):
        self.df = df
        self.resolver = resolver
        self.names = names

    def visit(self, node: ast.AST) -> Any:
        method = getattr(self, f"v_{type(node).__name__}")
        return method(node)

    def v_Constant(self, node: ast.Constant) -> Any:
        return node.value

    def v_List(self, node: ast.List) -> list[Any]:
        return [self.visit(e) for e in node.elts]

    v_Tuple = v_List

    def v_Name(self, node: ast.Name) -> Any:
        if node.id in _LITERALS:
            return _LITERALS[node.id]
        name = self.names.get(node.id, node.id)
        col = self.resolver.resolve(name)
        if col is None:
            raise ExpressionError(f"unknown column {name!r}")
        return self.df[col]

    def v_Call(self, node: ast.Call) -> Any:
        assert isinstance(node.func, ast.Name)
        return _FUNCS[node.func.id](*[self.visit(a) for a in node.args])

    def v_UnaryOp(self, node: ast.UnaryOp) -> Any:
        val = self.visit(node.operand)
        if isinstance(node.op, ast.Not):
            return ~_as_bool(val) if isinstance(val, pd.Series) else not val
        num = _to_num(val)
        return -num if isinstance(node.op, ast.USub) else num

    def v_BoolOp(self, node: ast.BoolOp) -> Any:
        vals = [_as_bool(self.visit(v)) for v in node.values]
        out = vals[0]
        for v in vals[1:]:
            out = (out & v) if isinstance(node.op, ast.And) else (out | v)
        return out

    def v_BinOp(self, node: ast.BinOp) -> Any:
        left, right = _to_num(self.visit(node.left)), _to_num(self.visit(node.right))
        with np.errstate(divide="ignore", invalid="ignore"):
            return _BIN[type(node.op)](left, right)

    def v_Compare(self, node: ast.Compare) -> Any:
        left = self.visit(node.left)
        result: Any = True
        for op, comp in zip(node.ops, node.comparators, strict=True):
            right = self.visit(comp)
            part = _compare(op, left, right)
            result = part if result is True else (_as_bool(result) & _as_bool(part))
            left = right
        return result


def _as_bool(v: Any) -> Any:
    if isinstance(v, pd.Series):
        if pd.api.types.is_bool_dtype(v):
            return v.fillna(False).astype(bool)
        return v.astype("object").where(v.notna(), False).astype(bool)
    return bool(v)


def _compare(op: ast.cmpop, left: Any, right: Any) -> Any:
    if isinstance(op, (ast.In, ast.NotIn)):
        if not isinstance(right, list):
            raise ExpressionError("`in` needs a list literal, e.g. q in ['a', 'b']")
        values = {str(v).strip() for v in right}
        nums = {float(v) for v in right if _is_numeric_like(v)}
        if isinstance(left, pd.Series):
            hit = _norm_str(left).isin(values)
            if nums:
                hit = hit | _to_num(left).isin(nums)
            return ~hit if isinstance(op, ast.NotIn) else hit
        found = str(left).strip() in values
        return (not found) if isinstance(op, ast.NotIn) else found
    fn = _CMP[type(op)]
    numeric = (_is_numeric_like(left) or left is None) and (_is_numeric_like(right) or right is None)
    if isinstance(left, pd.Series) and isinstance(right, pd.Series):
        numeric = _is_numeric_like(left) and _is_numeric_like(right)
    elif (
        isinstance(left, pd.Series) and isinstance(right, str) or isinstance(right, pd.Series) and isinstance(left, str)
    ):
        numeric = False
    if numeric and not (left is None or right is None):
        a, b = _to_num(left), _to_num(right)
        res = fn(a, b)
        if isinstance(res, pd.Series):
            valid = pd.Series(True, index=res.index)
            if isinstance(a, pd.Series):
                valid &= a.notna()
            if isinstance(b, pd.Series):
                valid &= b.notna()
            return res.astype(bool) & valid
        return bool(res) and not (pd.isna(a) or pd.isna(b))
    # string comparison (missing == "")
    a = _norm_str(left) if isinstance(left, pd.Series) else ("" if left is None else str(left).strip())
    b = _norm_str(right) if isinstance(right, pd.Series) else ("" if right is None else str(right).strip())
    if isinstance(op, (ast.Eq, ast.NotEq)):
        return fn(a, b)
    # ordering on strings: fall back to numeric when possible
    an, bn = _to_num(left), _to_num(right)
    res = fn(an, bn)
    return res.fillna(False).astype(bool) if isinstance(res, pd.Series) else bool(res)


def evaluate(expr: str, df: pd.DataFrame, resolver: ColumnResolver | None = None) -> pd.Series:
    return Expression(expr).evaluate(df, resolver)
