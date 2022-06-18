#!pip install -e /content/drive/MyDrive/Documenti/Scuola/Università/Sapienza/Tesi/EEG-connectivity-estimate-with-transformers/spacetimeformer/.
import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
import spacetimeformer as stf


from argparse import ArgumentParser
import sys
import seaborn as sns
import os

checkpoint_path = './data/stf_model_checkpoints/spatiotemporal_toy_eeg_468116566/spatiotemporal_toy_eegepoch=25-val/norm_mse=0.57.ckpt'


def create_parser():
    model = sys.argv[1]
    dset = sys.argv[2]

    parser = ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("dset")

    if dset=='toy2' or dset=='crypto':
        stf.data.CSVTimeSeries.add_cli(parser)
        stf.data.CSVTorchDset.add_cli(parser)
    elif dset=='toy_eeg':
        stf.data.EEGTimeSeries.add_cli(parser)
        stf.data.EEGTorchDset.add_cli(parser)
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


def create_model(config):
    x_dim, yc_dim, yt_dim = None, None, None
    if config.dset == "exchange":
        x_dim = 6
        yc_dim = 8
        yt_dim = 8
    elif config.dset == "toy2":
        x_dim = 6
        yc_dim = 20
        yt_dim = 20
    elif config.dset == "crypto":
        x_dim = 6
        yc_dim = 18
        yt_dim = 18
    elif config.dset == "toy_eeg":
        x_dim = 6
        yc_dim = 29
        yt_dim = 29
    assert x_dim is not None
    assert yc_dim is not None
    assert yt_dim is not None

    if config.model == "spacetimeformer":
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


def create_dset(config):
    INV_SCALER = lambda x: x
    SCALER = lambda x: x
    NULL_VAL = None
    PLOT_VAR_IDXS = None
    PLOT_VAR_NAMES = None
   
    if config.dset == "toy_eeg":
        data_path = "./data/toy_eeg/toy_eeg.pkl"

        dset = stf.data.EEGTimeSeries(
            data_path=data_path
        )
        print(f"config.context_points {config.context_points}")
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.EEGTorchDset,
            dataset_kwargs={
                "eeg_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
        )
        INV_SCALER = dset.reverse_scaling_eeg
        SCALER = dset.apply_scaling_eeg
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
    print(f"xc {xc}, yc {yc}, xt {xt}, yt {yt}")
    print(f"yt_pred {yt_pred}")

    # Plotting a prediction
    if not os.path.exists("./plots/"+checkpoint_path[29:59]):
        os.mkdir("./plots/"+checkpoint_path[29:59])

    plt.plot([*yc[0,:,0], *yt[0,:,0]])
    plt.plot([*yc[0,:,0], *yt_pred[0,:,0]])
    plt.savefig("./plots/"+checkpoint_path[29:59]+"/yc+yt.png")
    plt.show()

    plt.plot([*yc[0,:,1], *yt[0,:,1]])
    plt.plot([*yc[0,:,1], *yt_pred[0,:,1]])
    plt.savefig("./plots/"+checkpoint_path[29:59]+"/yc+yt.png")
    plt.show()

    plt.plot([*yc[0,:,2], *yt[0,:,2]])
    plt.plot([*yc[0,:,2], *yt_pred[0,:,2]])
    plt.savefig("./plots/"+checkpoint_path[29:59]+"/yc+yt.png")
    plt.show()

    plt.plot([*yc[0,:,3], *yt[0,:,3]])
    plt.plot([*yc[0,:,3], *yt_pred[0,:,3]])
    plt.savefig("./plots/"+checkpoint_path[29:59]+"/yc+yt.png")
    plt.show()


    plt.plot(yc[0,:,0])
    plt.savefig("./plots/"+checkpoint_path[29:59]+"/yc.png")
    plt.show()

    plt.plot(yt[0,:,0])
    plt.plot(yt_pred[0,:,0])
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