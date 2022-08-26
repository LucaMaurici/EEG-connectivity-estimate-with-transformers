#test_eeg

#!pip install -e /content/drive/MyDrive/Documenti/Scuola/Università/Sapienza/Tesi/EEG-connectivity-estimate-with-transformers/spacetimeformer/.
import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
import spacetimeformer as stf


from argparse import ArgumentParser
import sys
import seaborn as sns
import os
import numpy as np

#checkpoint_path = './data/stf_model_checkpoints/spatiotemporal_eeg_social_memory_734411140/spatiotemporal_eeg_social_memoryepoch=02-val/norm_mse=0.00.ckpt'
#checkpoint_path = './data/stf_model_checkpoints/spatiotemporal_eeg_social_memory_961033858/spatiotemporal_eeg_social_memoryepoch=32-val/norm_mse=0.00.ckpt'
#checkpoint_path = './data/stf_model_checkpoints/spatiotemporal_eeg_social_memory_573130492/spatiotemporal_eeg_social_memoryepoch=04-val/norm_mse=0.00.ckpt'
checkpoint_path = './data/stf_model_checkpoints/spatiotemporal_eeg_social_memory_991653722/spatiotemporal_eeg_social_memoryepoch=21-val/norm_mse=0.03.ckpt'

def create_parser():
    model = sys.argv[1]
    dset = sys.argv[2]

    parser = ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("dset")

    if dset=='toy2' or dset=='crypto':
        stf.data.CSVTimeSeries.add_cli(parser)
        stf.data.CSVTorchDset.add_cli(parser)
    elif dset=='toy_eeg' or dset=='eeg_social_memory':
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
        yc_dim = 5
        yt_dim = 5
    elif config.dset == 'eeg_social_memory':
        x_dim = 6
        yc_dim = 5
        yt_dim = 5
    assert x_dim is not None
    assert yc_dim is not None
    assert yt_dim is not None

    if config.model == "spacetimeformer":
        '''
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
        '''
        # new
        if hasattr(config, "context_points") and hasattr(config, "target_points"):
            max_seq_len = config.context_points + config.target_points
        elif hasattr(config, "max_len"):
            max_seq_len = config.max_len
        else:
            raise ValueError("Undefined max_seq_len")
        forecaster = stf.spacetimeformer_model.Spacetimeformer_Forecaster(
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            max_seq_len=max_seq_len,
            start_token_len=config.start_token_len,
            attn_factor=config.attn_factor,
            d_model=config.d_model,
            d_queries_keys=config.d_qk,
            d_values=config.d_v,
            n_heads=config.n_heads,
            e_layers=config.enc_layers,
            d_layers=config.dec_layers,
            d_ff=config.d_ff,
            dropout_emb=config.dropout_emb,
            dropout_attn_out=config.dropout_attn_out,
            dropout_attn_matrix=config.dropout_attn_matrix,
            dropout_qkv=config.dropout_qkv,
            dropout_ff=config.dropout_ff,
            pos_emb_type=config.pos_emb_type,
            use_final_norm=not config.no_final_norm,
            global_self_attn=config.global_self_attn,
            local_self_attn=config.local_self_attn,
            global_cross_attn=config.global_cross_attn,
            local_cross_attn=config.local_cross_attn,
            performer_kernel=config.performer_kernel,
            performer_redraw_interval=config.performer_redraw_interval,
            attn_time_windows=config.attn_time_windows,
            use_shifted_time_windows=config.use_shifted_time_windows,
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
            class_loss_imp=config.class_loss_imp,
            recon_loss_imp=config.recon_loss_imp,
            time_emb_dim=config.time_emb_dim,
            null_value=config.null_value,
            #pad_value=config.pad_value,  # mod commented
            linear_window=config.linear_window,
            use_revin=config.use_revin,
            linear_shared_weights=config.linear_shared_weights,
            use_seasonal_decomp=config.use_seasonal_decomp,
            use_val=not config.no_val,
            use_time=not config.no_time,
            use_space=not config.no_space,
            use_given=not config.no_given,
            recon_mask_skip_all=config.recon_mask_skip_all,
            recon_mask_max_seq_len=config.recon_mask_max_seq_len,
            recon_mask_drop_seq=config.recon_mask_drop_seq,
            recon_mask_drop_standard=config.recon_mask_drop_standard,
            recon_mask_drop_full=config.recon_mask_drop_full,
        )
    return forecaster


def create_dset(config):
    INV_SCALER = lambda x: x
    SCALER = lambda x: x
    NULL_VAL = None
    PLOT_VAR_IDXS = None
    PLOT_VAR_NAMES = None
   
    if 'eeg' in config.dset:
        if config.dset == "toy_eeg":
            data_path = "./data/toy_eeg/toy_eeg.pkl"
        elif config.dset == "eeg_social_memory":
            data_path = "./data/eeg_social_memory/eeg_social_memory_sub_20.pkl"

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
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        SCALER_WITH_DATA = dset.apply_scaling_with_data
        NULL_VAL = None

    return DATA_MODULE, INV_SCALER, SCALER, SCALER_WITH_DATA, NULL_VAL, PLOT_VAR_IDXS, PLOT_VAR_NAMES


def main(args):
    # Loading dataset
    (   data_module,
        inv_scaler,
        scaler,
        scaler_with_data,
        null_val,
        plot_var_idxs,
        plot_var_names,
    ) = create_dset(args)

    # Plotting dataset sample
    '''
    test_samples, test_samples_unscaled = next(iter(data_module.train_dataloader()))
    xc, yc_us, xt, yt_us = test_samples_unscaled
    #Statistical properties mod
    print(f"yc_us.shape: {yc_us.shape}")
    mean_yc_us = torch.mean(yc_us, dim=(0,1))
    std_dev_yc_us = torch.std(yc_us, dim=(0,1))
    #print(y_c[:,:,0])
    mean_yt_us = torch.mean(yt_us, dim=(0,1))
    std_dev_yt_us = torch.std(yt_us, dim=(0,1))
    print("\n\n-----------Statistical properties test_samples_unscaled------------")
    print(f"mean yc_us: {mean_yc_us}")
    print(f"std_dev yc_us: {std_dev_yc_us}")
    print()
    print(f"mean yt_us: {mean_yt_us}")
    print(f"std_dev yt_us: {std_dev_yt_us}")
    print("\n")

    # Loading model
    forecaster = create_model(args)
    forecaster.set_inv_scaler(inv_scaler)
    forecaster.set_scaler(scaler)
    forecaster.set_null_value(null_val)
    print(f'\n\n-------\ncheckpoint_path {checkpoint_path}\n\n-----------')
    checkpoint = torch.load(checkpoint_path)

    forecaster.load_state_dict(checkpoint['state_dict'])

    # Make a prediction
    yc, scaler_data = scaler(yc_us)
    yc = torch.from_numpy(yc).float()
    yt = scaler_with_data(yt_us, scaler_data)
    forecaster.to("cuda")
    yt_pred = forecaster.predict(xc, yc, xt)
    yc = inv_scaler(yc, scaler_data)
    yt = inv_scaler(yt, scaler_data)
    yt_pred = inv_scaler(yt_pred, scaler_data)
    print(f"xc {xc}, yc {yc}, xt {xt}, yt {yt}")
    print(f"yt_pred {yt_pred}")

    # Plotting a prediction
    if not os.path.exists("./plots/"+checkpoint_path[29:59]):
        os.mkdir("./plots/"+checkpoint_path[29:59])
        
    if type(yc).__module__ != np.__name__:  # not numpy
        yc = torch.Tensor.cpu(yc)
    if type(yt).__module__ != np.__name__:  # not numpy
        yt = torch.Tensor.cpu(yt)
    if type(yt_pred).__module__ != np.__name__:  # not numpy
        yt_pred = torch.Tensor.cpu(yt_pred)

    CHANNELS = 5
    for ch in range(CHANNELS):
        plt.plot([*yc[0,:,ch], *yt_pred[0,:,ch]])
        plt.plot([*yc[0,:,ch], *yt[0,:,ch]])
        plt.savefig(f"./plots/{checkpoint_path[29:59]}/context_and_predictions_ch_{ch}.png")
        plt.show()
    '''

    test_samples, test_samples_unscaled = next(iter(data_module.train_dataloader()))
    xc, yc, xt, yt = test_samples
    #Statistical properties mod

    # Loading model
    forecaster = create_model(args)
    #forecaster.set_inv_scaler(inv_scaler)
    #forecaster.set_scaler(scaler)
    #forecaster.set_null_value(null_val)
    print(f'\n\n-------\ncheckpoint_path {checkpoint_path}\n\n-----------')
    checkpoint = torch.load(checkpoint_path)

    forecaster.load_state_dict(checkpoint['state_dict'])

    # Make a prediction
    forecaster.to("cuda")
    yt_pred = forecaster.predict(xc, yc, xt)

    # Plotting a prediction
    if not os.path.exists("./plots/"+checkpoint_path[29:59]):
        os.mkdir("./plots/"+checkpoint_path[29:59])
        
    if type(yc).__module__ != np.__name__:  # not numpy
        yc = torch.Tensor.cpu(yc)
    if type(yt).__module__ != np.__name__:  # not numpy
        yt = torch.Tensor.cpu(yt)
    if type(yt_pred).__module__ != np.__name__:  # not numpy
        yt_pred = torch.Tensor.cpu(yt_pred)

    CHANNELS = 5
    for ch in range(CHANNELS):
        plt.plot([*yc[0,:,ch], *yt_pred[0,:,ch]])
        plt.plot([*yc[0,:,ch], *yt[0,:,ch]])
        plt.savefig(f"./plots/{checkpoint_path[29:59]}/context_and_predictions_ch_{ch}.png")
        plt.show()


    trainer = pl.Trainer(
        gpus=args.gpus,
        accelerator="auto"
    )

    # Test
    #trainer.test(model=forecaster, datamodule=data_module, ckpt_path=checkpoint_path)  # ckpt_path="best"
    trainer.test(model=forecaster, dataloaders=data_module.test_dataloader(), ckpt_path=checkpoint_path)  # ckpt_path="best"
    

if __name__ == "__main__":
    # CLI
    parser = create_parser()
    args = parser.parse_args()

    for trial in range(args.trials):
        main(args)
