import json

from naime_hybrid.training.logging_utils import JsonlMetricLogger, metrics_jsonl_to_csv


def test_jsonl_metric_logger_writes_async_and_closes(tmp_path):
    path = tmp_path / "metrics.jsonl"
    logger = JsonlMetricLogger(path, flush_every=4, fsync_every=0)
    for step in range(12):
        logger.write({"step": step, "lm": 1.0 / (step + 1)})
    logger.close()

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["step"] for row in rows] == list(range(12))


def test_jsonl_metric_logger_force_flush_and_csv(tmp_path):
    path = tmp_path / "metrics.jsonl"
    logger = JsonlMetricLogger(path, flush_every=100, fsync_every=0)
    logger.write({"step": 1, "lm": 2.5}, force_sync=True)
    assert path.read_text(encoding="utf-8").strip()
    logger.close()

    csv_path = metrics_jsonl_to_csv(path)
    assert csv_path is not None
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "step" in csv_text
    assert "lm" in csv_text
