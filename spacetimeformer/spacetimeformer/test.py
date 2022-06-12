import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
import spacetimeformer as stf

def create_dset():
    # Defining config
    DATASET = 'toy2' # eeg
    CONTEXT_POINTS = 128
    TARGET_POINTS = 32
    TIME_RESOLUTION = None
    BATCH_SIZE = 64
    WORKERS = None

    PLOT_VAR_IDXS = None
    PLOT_VAR_NAMES = None
    if DATASET == 'toy2':  
        data_path = './data/toy_dset2'
        checkpoint_path = './data/stf_model_checkpoints/spatiotemporal_toy2_568897978/spatiotemporal_toy2epoch=16-val/mse=0.71.ckpt'
        target_cols = [f"D{i}" for i in range(1, 21)]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols="all",
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif DATASET == 'eeg':
        data_path = './data/eeg'

    return DATA_MODULE, INV_SCALER, SCALER, NULL_VAL, PLOT_VAR_IDXS, PLOT_VAR_NAMES

# Loading dataset
(   data_module,
    inv_scaler,
    scaler,
    null_val,
    plot_var_idxs,
    plot_var_names,
) = create_dset()

# Plotting dataset sample
test_samples = next(iter(data_module.test_dataloader()))
xc, yc, xt, _ = test_samples
print(f"yc {yc.shape}, yt_pred {yt_pred.shape}, yt {yt.shape}")

# Loading model
forecaster = create_model(args)
forecaster.set_inv_scaler(inv_scaler)
forecaster.set_scaler(scaler)
forecaster.set_null_value(null_val)
checkpoint = torch.load(checkpoint_path)
forecaster.load_state_dict(checkpoint['model_state_dict'])

# Make a prediction
forecaster.to("cuda")
yt_pred = forecaster.predict(xc, yc, xt)

# Plotting a prediction
#plt.plot(yc)