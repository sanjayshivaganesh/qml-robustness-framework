"""Benchmark framework package.

Model adapters live in ``benchmark.adapters``. Pattern batches and the two
corruption procedures live in ``benchmark.data``. Metric selection lives in
``benchmark.scoring``. The noise-sweep smoke runner lives in
``benchmark.runner``. Request files are read by ``benchmark.config_loader``.
Procedure ids are resolved by ``benchmark.registry``.
``python -m benchmark`` validates or runs a request file.
Importing these loads the existing helpers they delegate to. It does not
run an experiment ``main`` and it does not write under ``results/``.
"""
