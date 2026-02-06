"""Integration tests for CLI."""

import pytest
from pathlib import Path
from click.testing import CliRunner

from taxonomise.cli import cli


@pytest.fixture
def runner():
    """Create CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_output(tmp_path):
    """Create temp output path."""
    return tmp_path / "output.csv"


class TestClassifyCommand:
    """Tests for the classify command."""

    def test_help(self, runner):
        """Test that help works."""
        result = runner.invoke(cli, ["classify", "--help"])
        assert result.exit_code == 0
        assert "Classify documents against a taxonomy" in result.output

    def test_missing_corpus(self, runner, fixtures_dir, temp_output):
        """Test error when corpus file is missing."""
        result = runner.invoke(
            cli,
            [
                "classify",
                "-c", "nonexistent.csv",
                "-t", str(fixtures_dir / "sample_taxonomy.csv"),
                "-o", str(temp_output),
            ],
        )
        assert result.exit_code != 0

    def test_missing_taxonomy(self, runner, fixtures_dir, temp_output):
        """Test error when taxonomy file is missing."""
        result = runner.invoke(
            cli,
            [
                "classify",
                "-c", str(fixtures_dir / "sample_corpus.csv"),
                "-t", "nonexistent.csv",
                "-o", str(temp_output),
            ],
        )
        assert result.exit_code != 0


# Note: Full integration tests that run the pipeline are commented out
# because they require downloading models which may be slow in CI.
#
# class TestFullPipeline:
#     """Full end-to-end tests."""
#
#     def test_classify_csv_to_csv(self, runner, fixtures_dir, temp_output):
#         """Test full classification from CSV to CSV."""
#         result = runner.invoke(
#             cli,
#             [
#                 "classify",
#                 "-c", str(fixtures_dir / "sample_corpus.csv"),
#                 "-t", str(fixtures_dir / "sample_taxonomy.csv"),
#                 "-o", str(temp_output),
#                 "-f", "csv",
#                 "--disable-keywords",
#                 "--disable-zeroshot",
#                 "-q",
#             ],
#         )
#         assert result.exit_code == 0
#         assert temp_output.exists()
