from river import evaluate, forest, metrics
from utils.dataset_utils import (
    HyperplaneStream,
    LabelShiftDataStream,
    SyntheticDataStream,
    ElectricityDataStream,
    AirlinesDataStream
)
import pandas as pd
from river import drift
import numpy as np

def ARF_run(seed, ds_type, ds_params, ddms):
    results = {}
    for ddm_name, drift_detectors in ddms.items():
        dataset = ds_type(**ds_params).get_data_stream()
        model = forest.ARFClassifier(seed=seed, leaf_prediction="mc", drift_detector=drift_detectors[0], warning_detector=drift_detectors[1])
        metric = metrics.Accuracy()
        results[ddm_name] = evaluate.progressive_val_score(dataset, model, metric).get()
    return results

def ARF_Experiment():
    print("Processing Label Shift...")
    label_shift_arf = ARF_run(
        seed=200,
        ds_type=LabelShiftDataStream,
        ds_params={
            "n":10000, 
            "ratios": [0.2, 0.5, 0.8, 0.5, 0.2],
            "seed": 200
        },
        ddms={
            "JSWIN": drift.JSWIN(alpha=0.45),
            "KSWIN": drift.KSWIN(alpha=0.001),
            "ADWIN": drift.ADWIN(delta=0.002)
        }
    )

    print("Processing Gaussian...")
    gauss_arf = ARF_run(
        seed=200,
        ds_type=SyntheticDataStream,
        ds_params={
            "distribution_types":[np.random.normal for i in range(5)],
            "pos_distribution_params": [(0, 2), (0.5, 3), (1, 2), (0.5, 3), (0, 2)],
            "neg_distribution_params": [(1, 2), (1.5, 3), (2, 2), (1.5, 3), (1, 2)],
            "samples_lens": [1000 for _ in range(5)],
            "seed": 200
        },
        ddms={
            "JSWIN": drift.JSWIN(alpha=0.45),
            "KSWIN": drift.KSWIN(alpha=0.001),
            "ADWIN": drift.ADWIN(delta=0.002)
        }
    )

    print("Processing Hyperplane...")
    hyperplane_arf = ARF_run(
        seed=200,
        ds_type=HyperplaneStream,
        ds_params={
            "n_drift_features": 2,
            "n_features": 2,
            "mag_change": 0.3,
            "seed": 200
        },
        ddms={
            "JSWIN": drift.JSWIN(alpha=0.45),
            "KSWIN": drift.KSWIN(alpha=0.001),
            "ADWIN": drift.ADWIN(delta=0.002)
        }
    )

    print("Processing Electricity...")
    electricity_arf = ARF_run(
        seed=200,
        ds_type=ElectricityDataStream,
        ds_params={},
        ddms={
            "JSWIN": [drift.JSWIN(alpha=0.45), drift.JSWIN(alpha=0.3)],
            "KSWIN": [drift.KSWIN(alpha=0.001), drift.KSWIN(alpha=0.01)],
            "ADWIN": [drift.ADWIN(delta=0.002), drift.ADWIN(delta=0.02)]
        }
    )

    print("Processing Airlines...")
    airlines_arf = ARF_run(
        seed=200,
        ds_type=AirlinesDataStream,
        ds_params={},
        ddms={
            "JSWIN": [drift.JSWIN(alpha=0.45), drift.JSWIN(alpha=0.3)],
            "KSWIN": [drift.KSWIN(alpha=0.001), drift.KSWIN(alpha=0.01)],
            "ADWIN": [drift.ADWIN(delta=0.002), drift.ADWIN(delta=0.02)]
        }
    )

    print(electricity_arf)
    print(airlines_arf)
    datasets = ['Hyperplane', 'Label Shift', 'Gaussian', 'Electricity']

    data = {
        'JSWIN': [hyperplane_arf['JSWIN'], label_shift_arf['JSWIN'], gauss_arf['JSWIN'], electricity_arf['JSWIN'], airlines_arf['JSWIN']],
        'KSWIN': [hyperplane_arf['KSWIN'], label_shift_arf['KSWIN'], gauss_arf['KSWIN'], electricity_arf['KSWIN'], airlines_arf['KSWIN']],
        'ADWIN': [hyperplane_arf['ADWIN'], label_shift_arf['ADWIN'], gauss_arf['ADWIN'], electricity_arf['ADWIN'], airlines_arf['ADWIN']]
    }

    return pd.DataFrame(data, index=datasets)