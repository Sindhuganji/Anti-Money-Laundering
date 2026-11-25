import json

import pandas as pd

from src import explore_data


def test_exploration_summary_includes_full_monthly_and_currency_counts(
    tmp_path,
    monkeypatch,
):
    data = pd.DataFrame(
        {
            "Timestamp": [
                "2022/01/01 00:00",
                "2022/01/15 12:00",
                "2022/02/01 00:00",
            ],
            "From Bank": [1, 2, 3],
            "Account": ["a", "b", "c"],
            "To Bank": [4, 5, 6],
            "Account.1": ["d", "e", "f"],
            "Amount Received": [10.0, 20.0, 30.0],
            "Receiving Currency": ["USD", "EUR", "USD"],
            "Amount Paid": [10.0, 20.0, 30.0],
            "Payment Currency": ["USD", "EUR", "USD"],
            "Payment Format": ["Wire", "Cash", "Wire"],
            "Is Laundering": [0, 1, 0],
        }
    )
    dataset_path = tmp_path / "transactions.csv"
    data.to_csv(dataset_path, index=False)
    monkeypatch.setattr(explore_data, "PLOT_DIR", tmp_path / "plots")
    monkeypatch.setattr(explore_data, "REPORT_DIR", tmp_path / "reports")

    summary = explore_data.run_exploration(dataset_path)

    assert summary["rows"] == 3
    assert summary["target_counts"] == {"0": 2, "1": 1}
    assert summary["volume_by_month"] == {
        "2022-01": 2,
        "2022-02": 1,
    }
    assert summary["laundering_volume_by_month"] == {"2022-01": 1}
    assert summary["receiving_currency_counts"] == {"USD": 2, "EUR": 1}
    assert summary["payment_currency_counts"] == {"USD": 2, "EUR": 1}
    assert "amount_received_log_distribution" in summary
    assert "numeric_correlations" in summary
    assert json.loads(
        (tmp_path / "reports" / "eda_summary.json").read_text()
    ) == summary
