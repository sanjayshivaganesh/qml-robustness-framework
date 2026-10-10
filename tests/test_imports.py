"""Import and path checks. Does not run the scientific experiments."""

import importlib
import inspect

import pytest


SRC_EXPORTS = {
    "src.hopfield": [
        "Hopfield",
        "HopfieldAsyncRandom",
        "hopfield_hebbian_weights",
        "hopfield_async_sequential_recall",
    ],
    "src.qam": [
        "qam_ansatz",
        "build_ry_linear_circuit",
        "train_ry_linear_qam",
        "train_ry_linear_qam_record_loss",
        "qam_recall",
        "nearest_neighbor_recall",
    ],
    "src.data_generation": [
        "random_bipolar_patterns",
        "corrupt_bernoulli",
        "corrupt_fixed_count",
    ],
    "src.metrics": [
        "bit_accuracy",
        "full_accuracy",
        "compute_metrics",
        "full_and_bitwise_accuracy",
        "pattern_accuracy_pair",
        "mean_std_ci",
        "bootstrap_mean_ci",
        "cliffs_delta",
        "cliffs_delta_magnitude",
    ],
    "src.paths": ["PROJECT_ROOT", "experiment_dir", "results_subdir"],
}

EXPERIMENT_MODULES = [
    "experiments.noise_robustness_sweep",
    "experiments.noise_robustness_inference",
    "experiments.storage_capacity",
    "experiments.scaling",
    "experiments.circuit_depth",
    "experiments.convergence",
    "experiments.statistical_significance",
]

SCRIPT_MODULES = [
    "scripts.plot_noise_robustness_sweep",
    "scripts.plot_noise_robustness_inference",
    "scripts.plot_scaling",
    "scripts.plot_inference_paper_figures",
    "scripts.plot_critical_noise_threshold",
    "scripts.cliffs_delta",
    "scripts.plot_bootstrap_distribution",
    "scripts.run_all_experiments",
]


@pytest.mark.parametrize("module_name, names", list(SRC_EXPORTS.items()))
def test_src_exports_exist(module_name, names):
    module = importlib.import_module(module_name)
    for name in names:
        assert hasattr(module, name), f"{module_name} is missing {name}"


def test_src_qam_has_no_qam_class():
    """Shared QAM code is functions, not a QAM class."""
    module = importlib.import_module("src.qam")
    assert not hasattr(module, "QAM")
    assert not hasattr(module, "QuantumAssociativeMemory")


@pytest.mark.parametrize("module_name", EXPERIMENT_MODULES + SCRIPT_MODULES)
def test_entry_point_imports(module_name):
    module = importlib.import_module(module_name)
    assert inspect.isfunction(getattr(module, "main", None)) or module_name.startswith("src")


@pytest.mark.parametrize("module_name", EXPERIMENT_MODULES + SCRIPT_MODULES)
def test_no_sys_path_hacks(module_name):
    module = importlib.import_module(module_name)
    source = inspect.getsource(module)
    assert "sys.path.insert" not in source
    assert "sys.path.append" not in source
