import os
import tempfile
from kognios.tools.builtins.calculator import calculator
from kognios.tools.builtins.read_file import read_file
from kognios.tools.builtins.write_file import write_file
from kognios.tools.builtins.python_eval import python_eval


def test_calculator_basic():
    assert calculator("2 + 3") == "5"
    assert calculator("10 * 4") == "40"
    assert calculator("2 ** 8") == "256"


def test_calculator_math_functions():
    result = calculator("sqrt(16)")
    assert result == "4.0"


def test_calculator_complex():
    result = calculator("round(3.14159, 2)")
    assert result == "3.14"


def test_calculator_invalid():
    result = calculator("import os")
    assert "Error" in result


def test_calculator_division():
    assert calculator("10 / 4") == "2.5"
    assert calculator("10 // 4") == "2"


def test_read_write_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("hello kognios")
        tmp = f.name
    try:
        content = read_file(tmp)
        assert content == "hello kognios"
    finally:
        os.unlink(tmp)


def test_read_file_missing():
    result = read_file("/nonexistent/path/file.txt")
    assert "Error" in result or "not found" in result


def test_write_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "sub", "file.txt")
        result = write_file(path, "test content")
        assert "Written" in result
        assert open(path).read() == "test content"


def test_python_eval_output():
    result = python_eval("print('hello')")
    assert result.strip() == "hello"


def test_python_eval_error():
    result = python_eval("raise ValueError('oops')")
    assert "Error" in result
