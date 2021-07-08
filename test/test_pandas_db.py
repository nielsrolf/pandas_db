from pandas_db import PandasDB
import tempfile


def test_db() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = PandasDB(tmp_dir)
        db.save(foo="foo", bar="bar")
        db.save(foo="FOO", another_column=42)
        df = db.get_df()
        assert set(df.columns) == set(['foo', 'bar', 'pandas_db.created', 'another_column'])
        assert df.loc[df.foo=="foo"].bar[0] == "bar"
        assert df.loc[df.foo=="foo"].another_column.isnull()[0]


def test_context() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = PandasDB(tmp_dir)
        with db.set_context(author="Alice"):
            with db.set_context(hello="world"):
                db.save(foo="foo", bar="bar")
        db.save(foo="FOO", another_column=42)
        df = db.get_df()
        assert set(df.columns) == set(['foo', 'bar', 'pandas_db.created', 'another_column', 'author', 'hello'])
        assert df.loc[df.foo=="foo"].bar[0] == "bar"
        assert df.loc[df.foo=="foo"].author[0] == "Alice"
        assert df.loc[df.foo=="foo"].hello[0] == "world"
        assert df.loc[df.foo=="FOO"].author.isnull()[0]
        assert df.loc[df.foo=="foo"].another_column.isnull()[0]