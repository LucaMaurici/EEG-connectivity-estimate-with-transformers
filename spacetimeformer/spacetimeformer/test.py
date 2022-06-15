#!pip install -e /content/drive/MyDrive/Documenti/Scuola/Università/Sapienza/Tesi/EEG-connectivity-estimate-with-transformers/spacetimeformer/.
import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
import spacetimeformer as stf


from argparse import ArgumentParser
import sys
import seaborn as sns
import os

DATASET = "toy2" # eeg
MODEL = "spacetimeformer"
#checkpoint_path = './data/stf_model_checkpoints/spatiotemporal_toy2_935412254/spatiotemporal_toy2epoch=18-val/norm_mse=0.19.ckpt'
checkpoint_path = './data/stf_model_checkpoints/spatiotemporal_toy2_658193677/spatiotemporal_toy2epoch=24-val/norm_mse=0.13.ckpt'

def create_parser():
    model = sys.argv[1]
    dset = sys.argv[2]

    parser = ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("dset")

    if dset=='toy2':
        stf.data.CSVTimeSeries.add_cli(parser)
        stf.data.CSVTorchDset.add_cli(parser)
    stf.data.DataModule.add_cli(parser)

    if model == "spacetimeformer":
        stf.spacetimeformer_model.Spacetimeformer_Forecaster.add_cli(parser)

    stf.callbacks.TimeMaskedLossCallback.add_cli(parser)

    parser.add_argument("--null_value", type=float, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--attn_plot", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--val_check_interval", type=float, default=1.0)
    parser.add_argument("--limit_val_batches", type=float, default=1.0)
    parser.add_argument(
        "--trials", type=int, default=1, help="How many consecutive trials to run"
    )

    if len(sys.argv) > 3 and sys.argv[3] == "-h":
        parser.print_help()
        sys.exit(0)

    return parser

'''
def create_model_old():
    x_dim, yc_dim, yt_dim = None, None, None
    if DATASET == "exchange":
        x_dim = 6
        yc_dim = 8
        yt_dim = 8
    elif DATASET == "toy2":
        x_dim = 6
        yc_dim = 20
        yt_dim = 20
    elif DATASET == "crypto":
        x_dim = 6
        yc_dim = 18
        yt_dim = 18
    assert x_dim is not None
    assert yc_dim is not None
    assert yt_dim is not None

    if MODEL == "spacetimeformer":
        #--d_model 10 --d_ff 40 --enc_layers 4 --dec_layers 4 --batch_size 7 --start_token_len 4 --n_heads 4
        D_MODEL = 10
        D_FF = 40
        ENC_LAYERS = 4
        DEC_LAYERS = 4
        BATCH_SIZE = 7
        START_TOKEN_LEN = 4
        N_HEADS = 4
        forecaster = stf.spacetimeformer_model.Spacetimeformer_Forecaster(
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            start_token_len=START_TOKEN_LEN,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            e_layers=ENC_LAYERS,
            d_layers=DEC_LAYERS,
            d_ff=D_FF,
        )
        
        forecaster = stf.spacetimeformer_model.Spacetimeformer_Forecaster(
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            start_token_len=config.start_token_len,
            attn_factor=config.attn_factor,
            d_model=config.d_model,
            n_heads=config.n_heads,
            e_layers=config.enc_layers,
            d_layers=config.dec_layers,
            d_ff=config.d_ff,
            dropout_emb=config.dropout_emb,
            dropout_token=config.dropout_token,
            dropout_attn_out=config.dropout_attn_out,
            dropout_qkv=config.dropout_qkv,
            dropout_ff=config.dropout_ff,
            global_self_attn=config.global_self_attn,
            local_self_attn=config.local_self_attn,
            global_cross_attn=config.global_cross_attn,
            local_cross_attn=config.local_cross_attn,
            performer_kernel=config.performer_kernel,
            performer_redraw_interval=config.performer_redraw_interval,
            post_norm=config.post_norm,
            norm=config.norm,
            activation=config.activation,
            init_lr=config.init_lr,
            base_lr=config.base_lr,
            warmup_steps=config.warmup_steps,
            decay_factor=config.decay_factor,
            initial_downsample_convs=config.initial_downsample_convs,
            intermediate_downsample_convs=config.intermediate_downsample_convs,
            embed_method=config.embed_method,
            l2_coeff=config.l2_coeff,
            loss=config.loss,
            linear_window=config.linear_window,
            class_loss_imp=config.class_loss_imp,
            time_emb_dim=config.time_emb_dim,
            null_value=config.null_value,
        )
        
    return forecaster
'''

def create_model(config):
    x_dim, yc_dim, yt_dim = None, None, None
    if DATASET == "exchange":
        x_dim = 6
        yc_dim = 8
        yt_dim = 8
    elif DATASET == "toy2":
        x_dim = 6
        yc_dim = 20
        yt_dim = 20
    elif DATASET == "crypto":
        x_dim = 6
        yc_dim = 18
        yt_dim = 18
    assert x_dim is not None
    assert yc_dim is not None
    assert yt_dim is not None

    if MODEL == "spacetimeformer":
        forecaster = stf.spacetimeformer_model.Spacetimeformer_Forecaster(
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            start_token_len=config.start_token_len,
            attn_factor=config.attn_factor,
            d_model=config.d_model,
            n_heads=config.n_heads,
            e_layers=config.enc_layers,
            d_layers=config.dec_layers,
            d_ff=config.d_ff,
            dropout_emb=config.dropout_emb,
            dropout_token=config.dropout_token,
            dropout_attn_out=config.dropout_attn_out,
            dropout_qkv=config.dropout_qkv,
            dropout_ff=config.dropout_ff,
            global_self_attn=config.global_self_attn,
            local_self_attn=config.local_self_attn,
            global_cross_attn=config.global_cross_attn,
            local_cross_attn=config.local_cross_attn,
            performer_kernel=config.performer_kernel,
            performer_redraw_interval=config.performer_redraw_interval,
            post_norm=config.post_norm,
            norm=config.norm,
            activation=config.activation,
            init_lr=config.init_lr,
            base_lr=config.base_lr,
            warmup_steps=config.warmup_steps,
            decay_factor=config.decay_factor,
            initial_downsample_convs=config.initial_downsample_convs,
            intermediate_downsample_convs=config.intermediate_downsample_convs,
            embed_method=config.embed_method,
            l2_coeff=config.l2_coeff,
            loss=config.loss,
            linear_window=config.linear_window,
            class_loss_imp=config.class_loss_imp,
            time_emb_dim=config.time_emb_dim,
            null_value=config.null_value,
        )
    return forecaster

'''
def create_dset_old():
    # Defining config
    # Old config
    CONTEXT_POINTS = 128
    TARGET_POINTS = 32
    TIME_RESOLUTION = 1
    BATCH_SIZE = 64
    WORKERS = 1

    PLOT_VAR_IDXS = None
    PLOT_VAR_NAMES = None
    if DATASET == 'toy2':  
        data_path = './data/toy_dset2.csv'
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
                "split": 'test',
                "context_points": CONTEXT_POINTS,
                "target_points": TARGET_POINTS,
                "time_resolution": TIME_RESOLUTION,
            },
            batch_size=BATCH_SIZE,
            workers=WORKERS,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif DATASET == 'eeg':
        data_path = './data/eeg'

    return DATA_MODULE, INV_SCALER, SCALER, NULL_VAL, PLOT_VAR_IDXS, PLOT_VAR_NAMES
'''

def create_dset(config):
    INV_SCALER = lambda x: x
    SCALER = lambda x: x
    NULL_VAL = None
    PLOT_VAR_IDXS = None
    PLOT_VAR_NAMES = None
   
    data_path = config.data_path
    if config.dset == "exchange":
        if data_path == "auto":
            data_path = "./data/exchange_rate_converted.csv"
        target_cols = [
            "Australia",
            "United Kingdom",
            "Canada",
            "Switzerland",
            "China",
            "Japan",
            "New Zealand",
            "Singapore",
        ]
    elif config.dset == "crypto":
        if data_path == "auto":
            data_path = "./data/crypto_dset.csv"
        target_cols = [
            "ETH_open",
            "ETH_high",
            "ETHT_low",
            "ETH_close",
            "Volume BTC",
            "Volume USDT",
            "ETH_tradecount",
            "BTC_open",
            "BTC_high",
            "BTC_low",
            "BTC_close",
            "BTC_tradecount",
            "LTCUSDT_open",
            "LTCUSDT_high",
            "LTCUSDT_low",
            "LTCUSDT_close",
            "Volume LTC",
            "LTCUSDT_tradecount",
        ]
        # only make plots of a few vars
        PLOT_VAR_NAMES = ["ETH_close", "BTC_close", "ETH_high", "BTC_high"]
        PLOT_VAR_IDXS = [target_cols.index(x) for x in PLOT_VAR_NAMES]
    elif config.dset == "toy2":
        data_path = './data/toy_dset2.csv'
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

    return DATA_MODULE, INV_SCALER, SCALER, NULL_VAL, PLOT_VAR_IDXS, PLOT_VAR_NAMES


def main(args):
    # Loading dataset
    (   data_module,
        inv_scaler,
        scaler,
        null_val,
        plot_var_idxs,
        plot_var_names,
    ) = create_dset(args)

    # Plotting dataset sample
    test_samples = next(iter(data_module.test_dataloader()))
    xc, yc, xt, yt = test_samples

    # Loading model
    forecaster = create_model(args)
    forecaster.set_inv_scaler(inv_scaler)
    forecaster.set_scaler(scaler)
    forecaster.set_null_value(null_val)
    print(f'\n\n-------\ncheckpoint_path {checkpoint_path}\n\n-----------')
    checkpoint = torch.load(checkpoint_path)
    '''
    print('\n--------CHECKPOINT KEYS----------')
    for key in checkpoint.keys():
        print(key)
    '''
    forecaster.load_state_dict(checkpoint['state_dict'])

    # Make a prediction
    forecaster.to("cuda")
    yt_pred = forecaster.predict(xc, yc, xt)
    print(f"xc {xc.shape}, yc {yc.shape}, xt {xt.shape}, yt {yt.shape}")
    print(f"yt_pred {yt_pred.shape}")

    # Plotting a prediction
    os.mkdir("./plots/"+checkpoint_path[29:59])
    plt.plot(yc[0,:,1:5])
    plt.savefig("./plots/"+checkpoint_path[29:59]+"/yc.png")
    plt.show()

    plt.plot(yt[0,:,1:5])
    plt.plot(yt_pred[0,:,1:5])
    plt.savefig("./plots/"+checkpoint_path[29:59]+"/yt_and_yt_pred.png")
    plt.show()

    '''
    sns.lineplot(data=yc[0,:,1:5])
    plt.savefig("./plots/yc_sns.png")
    plt.show()
    '''


if __name__ == "__main__":
    # CLI
    parser = create_parser()
    args = parser.parse_args()

    for trial in range(args.trials):
        main(args)