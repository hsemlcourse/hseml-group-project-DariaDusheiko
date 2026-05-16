"""
Патч для загрузки pickle-файлов, созданных в pandas < 2.0,
в окружениях с pandas >= 2.0.

Использование: импортируйте ДО import pandas, или сразу после:
    import pandas_compat_patch
    import pandas as pd
"""
import sys
import types
import pandas as pd

# Создаём фиктивный модуль pandas.core.indexes.numeric
_mod = types.ModuleType("pandas.core.indexes.numeric")


class Int64Index(pd.Index):
    def __new__(cls, data=None, dtype=None, copy=False, name=None, **kwargs):
        return pd.Index(data, dtype="int64", copy=copy, name=name)


class Float64Index(pd.Index):
    def __new__(cls, data=None, dtype=None, copy=False, name=None, **kwargs):
        return pd.Index(data, dtype="float64", copy=copy, name=name)


class UInt64Index(pd.Index):
    def __new__(cls, data=None, dtype=None, copy=False, name=None, **kwargs):
        return pd.Index(data, dtype="uint64", copy=copy, name=name)


class NumericIndex(pd.Index):
    def __new__(cls, data=None, dtype=None, copy=False, name=None, **kwargs):
        return pd.Index(data, dtype=dtype, copy=copy, name=name)


_mod.Int64Index = Int64Index
_mod.Float64Index = Float64Index
_mod.UInt64Index = UInt64Index
_mod.NumericIndex = NumericIndex

sys.modules["pandas.core.indexes.numeric"] = _mod
