"""CLI routing tests with mocked backends: no model downloads, GPU, or MLX needed."""

import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call, create_autospec

import pytest

import semif_phase1
from semif_phase1 import cli
from semif_phase1.cli import main


@pytest.mark.parametrize("extra,message", [
    (["--backend", "mlx", "--mode", "reranker"], "reranker requires torch"),
    (["--mode", "direct", "--mlx-bits", "4"], "requires --backend mlx"),
    (["--mode", "direct", "--mlx-cache-limit-mib", "0"], "requires --backend mlx"),
    (["--mode", "direct", "--backend", "mlx", "--mlx-cache-limit-mib", "-1"], "must be nonnegative"),
    (["--mode", "direct", "--sglang-url", "http://127.0.0.1:30000"], "require --backend sglang"),
    (["--mode", "direct", "--sglang-timeout", "5"], "require --backend sglang"),
])
def test_invalid_backend_combinations_fail_before_loading(tmp_path, monkeypatch, capsys, extra, message):
    monkeypatch.setattr(sys, "argv", ["semif-score", "--model", "unused", "--revision", "unused",
                                    "--input", "missing.jsonl", "--output", str(tmp_path / "out.jsonl"), *extra])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "out.jsonl").exists()


@pytest.mark.parametrize('limit', [None, 0, 512])
def test_cli_passes_cache_limit_to_loader(tmp_path, monkeypatch, limit):
    fake_backend = SimpleNamespace(
        DEFAULT_CACHE_LIMIT_MIB=256,
        load_model=lambda source, revision, bits, *, cache_limit_mib:
            (None, None, {'limit': cache_limit_mib}),
        score=lambda model, tokenizer, row, metadata, max_tokens: metadata,
        SerialPrefixScorer=None, score_shared=None,
    )
    monkeypatch.setattr(semif_phase1, 'mlx_backend', fake_backend, raising=False)
    source, output = tmp_path / 'input.jsonl', tmp_path / 'output.jsonl'
    source.write_text(json.dumps({'id': 'test', 'state': 'Evidence', 'question': 'Supported?',
                                 'options': [{'id': 'yes', 'description': 'Yes'}, {'id': 'no', 'description': 'No'}]}) + '\n')
    args = ['semif-score', '--backend', 'mlx', '--mode', 'direct', '--model', 'unused',
            '--revision', 'unused', '--input', str(source), '--output', str(output)]
    if limit is not None:
        args += ['--mlx-cache-limit-mib', str(limit)]
    monkeypatch.setattr(sys, 'argv', args)
    main()
    assert json.loads(output.read_text())['limit'] == (256 if limit is None else limit)


@pytest.fixture
def run_cli(tmp_path, monkeypatch):
    rows = [
        {"id": key, "state": "shared evidence", "question": "Which answer follows?",
         "options": [{"id": "yes", "description": "Yes."}, {"id": "no", "description": "No."}]}
        for key in ("first", "second")
    ]
    source = tmp_path / "input.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    output = tmp_path / "nested" / "output.jsonl"

    def run(mode="direct", *flags):
        monkeypatch.setattr(sys, "argv", [
            "semif-score", "--mode", mode, "--model", "test/model",
            "--revision", "a" * 40, "--input", str(source), "--output", str(output),
            "--max-tokens", "128", *flags,
        ])
        cli.main()

    return SimpleNamespace(run=run, rows=rows, output=output)


@pytest.fixture
def backends(monkeypatch):
    loaded = (object(), object(), {"revision": "a" * 40})
    results = [{"id": "first", "probabilities": [0.25, 0.75]},
               {"id": "second", "probabilities": [0.8, 0.2]}]
    timing = {"total_seconds": 0.125}

    def mlx_loader(source, revision, bits, *, cache_limit_mib):
        """Real signature: stale torch-only arguments fail here."""

    mlx = ModuleType("semif_phase1.mlx_backend")
    mlx.DEFAULT_CACHE_LIMIT_MIB = 256
    mlx.load_model = create_autospec(mlx_loader, return_value=loaded)
    mlx.score = Mock(side_effect=results)
    mlx.score_shared = Mock(return_value=(results, timing))
    mlx.SerialPrefixScorer = Mock(return_value=SimpleNamespace(score=Mock(side_effect=results)))
    monkeypatch.setitem(sys.modules, mlx.__name__, mlx)
    monkeypatch.setattr(semif_phase1, "mlx_backend", mlx, raising=False)

    torch = SimpleNamespace(
        load_causal_model=Mock(return_value=loaded),
        direct_score=Mock(side_effect=results),
        score_shared=Mock(return_value=(results, timing)),
        SerialPrefixScorer=Mock(return_value=SimpleNamespace(score=Mock(side_effect=results))),
        reranker_score=Mock(side_effect=results),
    )
    for name, mock in vars(torch).items():
        monkeypatch.setattr(cli, name, mock)
    return SimpleNamespace(mlx=mlx, torch=torch, loaded=loaded, results=results, timing=timing)


@pytest.mark.parametrize("mode", ["direct", "serial", "shared"])
def test_mlx_routes_to_matching_scorer(run_cli, backends, mode):
    run_cli.run(mode, "--backend", "mlx")
    mlx = backends.mlx
    model, tokenizer, metadata = backends.loaded
    mlx.load_model.assert_called_once()
    assert all(mock.call_count == 0 for mock in vars(backends.torch).values())
    if mode == "direct":
        assert mlx.score.call_args_list == [call(model, tokenizer, row, metadata, 128) for row in run_cli.rows]
    elif mode == "serial":
        assert mlx.SerialPrefixScorer.return_value.score.call_args_list == [call(row) for row in run_cli.rows]
    else:
        mlx.score_shared.assert_called_once_with(model, tokenizer, run_cli.rows, metadata, 128)
    expected = backends.results if mode != "shared" else [
        {**result, "shared_timing": backends.timing} for result in backends.results
    ]
    assert [json.loads(line) for line in run_cli.output.read_text().splitlines()] == expected


@pytest.mark.parametrize("flags", [[], ["--device", "auto"], ["--device", "cuda"]])
def test_torch_reranker_always_loads_cuda(run_cli, backends, flags):
    run_cli.run("reranker", *flags)
    backends.torch.load_causal_model.assert_called_once_with("test/model", "a" * 40, "cuda", "bfloat16")
    model, tokenizer, metadata = backends.loaded
    assert backends.torch.reranker_score.call_args_list == [
        call(model, tokenizer, row, metadata, 128) for row in run_cli.rows
    ]
    backends.mlx.load_model.assert_not_called()


@pytest.mark.parametrize("device", ["mps", "cpu"])
def test_torch_reranker_rejects_non_cuda_device_before_loading(run_cli, backends, capsys, device):
    run_cli.run("reranker")
    run_cli.output.unlink()
    with pytest.raises(SystemExit) as error:
        run_cli.run("reranker", "--device", device)
    assert error.value.code == 2
    assert "Reranker mode requires CUDA" in capsys.readouterr().err
    assert backends.torch.reranker_score.call_count == 2


def test_torch_cpu_selection_reaches_loader(run_cli, backends):
    run_cli.run("direct", "--device", "cpu", "--dtype", "float32")
    backends.torch.load_causal_model.assert_called_once_with(
        "test/model", "a" * 40, "cpu", "float32")


def test_existing_output_is_not_overwritten(run_cli, backends, capsys):
    original = b"existing benchmark evidence\n"
    run_cli.output.parent.mkdir(parents=True)
    run_cli.output.write_bytes(original)
    with pytest.raises(SystemExit) as error:
        run_cli.run("direct")
    assert error.value.code == 2
    assert "Output must be new" in capsys.readouterr().err
    assert run_cli.output.read_bytes() == original
    backends.torch.load_causal_model.assert_not_called()
    backends.mlx.load_model.assert_not_called()


@pytest.fixture
def sglang(backends, monkeypatch):
    from semif_phase1 import sglang_backend

    fake = SimpleNamespace(
        load_model=create_autospec(sglang_backend.load_model, return_value=backends.loaded),
        score=Mock(side_effect=backends.results),
        SerialPrefixScorer=Mock(return_value=SimpleNamespace(score=Mock(side_effect=backends.results))),
        score_shared=Mock(return_value=(backends.results, backends.timing)),
        reranker_score=Mock(side_effect=backends.results),
    )
    for name, value in vars(fake).items():
        monkeypatch.setattr(sglang_backend, name, value)
    return fake


@pytest.mark.parametrize("mode", ["direct", "serial", "shared", "reranker"])
def test_sglang_routes_every_mode_to_its_scorer(run_cli, backends, sglang, mode):
    flags = ["--sglang-url", "http://127.0.0.1:30001", "--sglang-timeout", "12.5"] if mode == "reranker" else []
    run_cli.run(mode, "--backend", "sglang", *flags)
    model, tokenizer, metadata = backends.loaded
    url, timeout = ("http://127.0.0.1:30001", 12.5) if flags else ("http://127.0.0.1:30000", 300.0)
    sglang.load_model.assert_called_once_with("test/model", "a" * 40, url=url, timeout=timeout, max_tokens=128)
    assert all(mock.call_count == 0 for mock in vars(backends.torch).values())
    if mode in {"direct", "reranker"}:
        scorer = sglang.score if mode == "direct" else sglang.reranker_score
        assert scorer.call_args_list == [call(model, tokenizer, row, metadata, 128) for row in run_cli.rows]
    elif mode == "serial":
        sglang.SerialPrefixScorer.assert_called_once_with(model, tokenizer, metadata, 128)
        assert sglang.SerialPrefixScorer.return_value.score.call_args_list == [call(row) for row in run_cli.rows]
    else:
        sglang.score_shared.assert_called_once_with(model, tokenizer, run_cli.rows, metadata, 128)
    expected = backends.results if mode != "shared" else [
        {**result, "shared_timing": backends.timing} for result in backends.results
    ]
    assert [json.loads(line) for line in run_cli.output.read_text(encoding="utf-8").splitlines()] == expected
