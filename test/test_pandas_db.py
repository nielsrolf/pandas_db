from pandas_db import PandasDB
import tempfile


def test_db() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = PandasDB(tmp_dir)
        db.save(foo="foo", bar="bar")
        db.save(foo="FOO", another_column=42)
        df = db.get_df()
        assert all(df.columns == ['foo', 'bar', 'entry_created', 'another_column'])
        assert df.loc[df.foo=="foo"].bar[0] == "bar"
        assert df.loc[df.foo=="foo"].another_column.isnull()[0]
