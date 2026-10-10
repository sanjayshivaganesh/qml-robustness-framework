"""Hamming nearest-neighbour recall.

This is the procedure behind inference rows with ``model="qam"``. It is not
a variational circuit and it has no training method. ``recall`` returns the
stored row from ``src.qam.nearest_neighbor_recall``.
"""

from src.qam import nearest_neighbor_recall

from benchmark.identity import ProcedureIdentity, RngContract, procedure_description

implementation = nearest_neighbor_recall

identity = ProcedureIdentity(
    procedure_id="hamming_nearest",
    family="hamming_nearest",
    source="src.qam.nearest_neighbor_recall",
    capabilities=("recall",),
)

rng_contract = RngContract(
    stream="none",
    construction_reseeds=False,
    construction_draws=False,
    notes=(
        "Classical Hamming nearest neighbour. No circuit and no parameter draw. "
        "The inference CSV value model='qam' is a column label, not this "
        "procedure's family."
    ),
)


def recall(patterns, query):
    """Return ``patterns[argmin]`` from ``nearest_neighbor_recall``."""
    return nearest_neighbor_recall(patterns, query)


def describe(**hyperparameters):
    base = {
        "distance": "sum(patterns != query) / n_bits",
        "tie_break": "numpy.argmin, first minimum",
        "variational": False,
        "csv_model_label_in_inference_archive": "qam",
        "recall_returns": "stored pattern row",
    }
    base.update(hyperparameters)
    return procedure_description(identity, rng_contract, **base)
