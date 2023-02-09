#test_eeg_channels_loss

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

from spacetimeformer.spacetimeformer_model.utils.general_utils import *
import config_file as cf
cf.reset()

#checkpoint_path = './data/stf_model_checkpoints/spatiotemporal_eeg_social_memory_734411140/spatiotemporal_eeg_social_memoryepoch=02-val/norm_mse=0.00.ckpt'
#checkpoint_path = './data/stf_model_checkpoints/spatiotemporal_eeg_social_memory_961033858/spatiotemporal_eeg_social_memoryepoch=32-val/norm_mse=0.00.ckpt'
#checkpoint_path = './data/stf_model_checkpoints/spatiotemporal_eeg_social_memory_573130492/spatiotemporal_eeg_social_memoryepoch=04-val/norm_mse=0.00.ckpt'
#checkpoint_path = './data/stf_model_checkpoints/spatiotemporal_eeg_social_memory_991653722/prima_run_epoch=21-val/norm_mse=0.03.ckpt'

def create_parser():
    model = sys.argv[1]
    dset = sys.argv[2]

    parser = ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("dset")

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


def create_model(config, num_channels):
    x_dim, yc_dim, yt_dim = None, None, None
    
    if config.dset == 'eeg_social_memory':
        x_dim = cf.read("x_dim")  #4  # mod 6
        yc_dim = num_channels
        yt_dim = num_channels

    assert x_dim is not None
    assert yc_dim is not None
    assert yt_dim is not None

    if config.model == "spacetimeformer":
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


def create_dset(config, channel):
    INV_SCALER = lambda x: x
    SCALER = lambda x: x
    NULL_VAL = None
    PLOT_VAR_IDXS = None
    PLOT_VAR_NAMES = None
   
    if 'eeg' in config.dset:
        if config.dset == "toy_eeg":
            data_path = "./data/toy_eeg/toy_eeg.pkl"
        elif config.dset == "eeg_social_memory":
            #data_path = "./data/eeg_social_memory/eeg_social_memory_29chs_downsampled_sub_20.pkl"
            #data_path = f"./data/Generated EEG/eeg_generated_ch{channel}_sub_{cf.read('subject_id')}.pkl"
            RUN_NICKNAME = cf.read('run_nickname')
            SUBJECT_ID = cf.read('subject_id')
            CONDITION = cf.read('condition')
            #data_path = f"./data/eeg_social_memory/{RUN_NICKNAME}/ch{channel}_sub_{SUBJECT_ID}_cond_{CONDITION}.pkl"
            data_path = f"./data/eeg_hyperscanning/{RUN_NICKNAME}/ch{channel}_sub_{SUBJECT_ID}_cond_{CONDITION}.pkl"
            print(data_path)
        dset = stf.data.EEGTimeSeries(
            data_path=data_path
        )
        context_points = 10
        target_points = 1
        time_resolution = 1
        batch_size = 128
        workers = 2
        print(f"context_points {context_points}")
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.EEGTorchDset,
            dataset_kwargs={
                "eeg_time_series": dset,
                "context_points": context_points,
                "target_points": target_points,
                "time_resolution": time_resolution,
            },
            batch_size=batch_size,
            workers=workers,
        )

        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        SCALER_WITH_DATA = dset.apply_scaling_with_data
        NULL_VAL = None

    return DATA_MODULE, INV_SCALER, SCALER, SCALER_WITH_DATA, NULL_VAL, PLOT_VAR_IDXS, PLOT_VAR_NAMES


def main(args):
    # add
    cf.reset()
    num_channels = int(cf.read('num_channels'))
    CHANNEL_START = 0
    CHANNEL_END = num_channels
    for channel in range(CHANNEL_START, CHANNEL_END+1):
        write_run_id(channel)
        if channel == CHANNEL_START:
            num_channels -= 1
            cf.write('num_channels', num_channels)
        elif channel == CHANNEL_END:
            num_channels += 1
            cf.write('num_channels', num_channels)
            
        args.run_name, run_type, run_id = read_config_file()
        print(f"args.run_name: {args.run_name}")
        if run_type == 'test':
            run_type = 'granger_test'
        else:
            raise Exception("MYERROR: Unknown or invalid run type")
        plots_folder = f"./plots_checkpoints_logs/{args.run_name}/plots/{channel}/{run_type}"
        #checkpoint_path = './plots_checkpoints_logs/{args.run_name}/checkpoints/{run_id}/epoch=21-val/norm_mse=0.03.ckpt'
        print("File list: ", get_list_of_files(f'./plots_checkpoints_logs/{args.run_name}/checkpoints/{channel}/'))
        list_of_files = get_list_of_files(f'./plots_checkpoints_logs/{args.run_name}/checkpoints/{channel}/')
        index = index_of_not_substring(list_of_files, 'desktop.ini')
        print(index)
        #checkpoint_path = get_list_of_files(f'.\\plots_checkpoints_logs\\{args.run_name}\\checkpoints\\{channel}\\')[index]
        checkpoint_path = get_list_of_files(f'./plots_checkpoints_logs/{args.run_name}/checkpoints/{channel}/')[index]
        #checkpoint_path = get_list_of_files(f'./plots_checkpoints_logs/{args.run_name}/checkpoints/{channel}/')[0]
        print(f"checkpoint_path: {checkpoint_path}")

        # Loading dataset
        (   data_module,
            inv_scaler,
            scaler,
            scaler_with_data,
            null_val,
            plot_var_idxs,
            plot_var_names,
        ) = create_dset(args, channel)

        test_samples, test_samples_unscaled = next(iter(data_module.train_dataloader()))
        xc, yc, xt, yt = test_samples
        #Statistical properties mod

        # Loading model
        #forecaster = create_model(args)
        #forecaster.set_inv_scaler(inv_scaler)
        #forecaster.set_scaler(scaler)
        #forecaster.set_null_value(null_val)
        print(f'\n\n-------\ncheckpoint_path {checkpoint_path}\n\n-----------')
        #checkpoint = torch.load(checkpoint_path)

        forecaster = stf.spacetimeformer_model.Spacetimeformer_Forecaster.load_from_checkpoint(checkpoint_path)
        forecaster.set_inv_scaler(inv_scaler)  # add

        # AttentionMatrixCallback
        callbacks = []
        batch_size = 128
        #if args.wandb and args.model == "spacetimeformer" and args.attn_plot:
        if args.model == "spacetimeformer" and args.attn_plot:
            callbacks.append(
                stf.plot.AttentionMatrixCallback(
                    test_samples,
                    layer=0,
                    total_samples=min(1024, batch_size),  # mod 16
                )
            )
            callbacks[0].on_called(forecaster)

        '''
        # Make a prediction
        forecaster.to("cuda")
        yt_pred = forecaster.predict(xc, yc, xt)

        # Plotting a prediction
        #if not os.path.exists("./plots/"+checkpoint_path[29:59]):
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)
            
        if type(yc).__module__ != np.__name__:  # not numpy
            yc = torch.Tensor.cpu(yc)
        if type(yt).__module__ != np.__name__:  # not numpy
            yt = torch.Tensor.cpu(yt)
        if type(yt_pred).__module__ != np.__name__:  # not numpy
            yt_pred = torch.Tensor.cpu(yt_pred)

        for ch in range(num_channels):
            plt.plot([*yc[0,:,ch], *yt_pred[0,:,ch]])
            plt.plot([*yc[0,:,ch], *yt[0,:,ch]])
            plt.savefig(f"{plots_folder}/context_and_predictions_ch_{ch}.png")
            plt.show()
        '''

        trainer = pl.Trainer(
            callbacks=callbacks,
            gpus=args.gpus,
            accelerator="auto",
            num_sanity_val_steps=0
        )

        # Test
        #trainer.test(model=forecaster, datamodule=data_module, ckpt_path=checkpoint_path)  # ckpt_path="best"
        trainer.test(model=forecaster, dataloaders=data_module.train_dataloader(), ckpt_path=checkpoint_path)  # ckpt_path="best"
    

if __name__ == "__main__":
    # CLI
    parser = create_parser()
    args = parser.parse_args()

    for trial in range(args.trials):
        main(args)
