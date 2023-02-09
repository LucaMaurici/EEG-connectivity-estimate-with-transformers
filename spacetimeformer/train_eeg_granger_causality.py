#train_eeg

from argparse import ArgumentParser
import random
import sys
import warnings
import os

import pytorch_lightning as pl
import torch

warnings.filterwarnings("ignore", category=DeprecationWarning)

import spacetimeformer as stf

from spacetimeformer.spacetimeformer_model.utils.general_utils import *
import config_file as cf
cf.reset()

_MODELS = ["spacetimeformer", "mtgnn", "lstm", "lstnet", "linear", "s4"]

_DSETS = [
    "asos",
    "metr-la",
    "pems-bay",
    "exchange",
    "precip",
    "toy1",
    "toy2",
    "solar_energy",
    "mnist",
    "cifar",
    "copy",
    "crypto",
    "toy_eeg",
    "eeg_social_memory"
]


def create_parser():
    model = sys.argv[1]
    dset = sys.argv[2]

    # Throw error now before we get confusing parser issues
    assert (
        model in _MODELS
    ), f"Unrecognized model (`{model}`). Options include: {_MODELS}"
    assert dset in _DSETS, f"Unrecognized dset (`{dset}`). Options include: {_DSETS}"

    parser = ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("dset")

    if dset == "precip":
        stf.data.precip.GeoDset.add_cli(parser)
        stf.data.precip.CONUS_Precip.add_cli(parser)
    elif dset == "metr-la" or dset == "pems-bay":
        stf.data.metr_la.METR_LA_Data.add_cli(parser)
    elif dset == "mnist":
        stf.data.image_completion.MNISTDset.add_cli(parser)
    elif dset == "cifar":
        stf.data.image_completion.CIFARDset.add_cli(parser)
    elif dset == "copy":
        stf.data.copy_task.CopyTaskDset.add_cli(parser)
    else:
        stf.data.EEGTimeSeries.add_cli(parser)
        stf.data.EEGTorchDset.add_cli(parser)
    stf.data.DataModule.add_cli(parser)

    if model == "lstm":
        stf.lstm_model.LSTM_Forecaster.add_cli(parser)
        stf.callbacks.TeacherForcingAnnealCallback.add_cli(parser)
    elif model == "lstnet":
        stf.lstnet_model.LSTNet_Forecaster.add_cli(parser)
    elif model == "mtgnn":
        stf.mtgnn_model.MTGNN_Forecaster.add_cli(parser)
    elif model == "spacetimeformer":
        stf.spacetimeformer_model.Spacetimeformer_Forecaster.add_cli(parser)
    elif model == "linear":
        stf.linear_model.Linear_Forecaster.add_cli(parser)
    elif model == "s4":
        stf.s4_model.S4_Forecaster.add_cli(parser)

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
    if config.dset == "metr-la":
        x_dim = 2
        yc_dim = 207
        yt_dim = 207
    elif config.dset == "pems-bay":
        x_dim = 2
        yc_dim = 325
        yt_dim = 325
    elif config.dset == "precip":
        x_dim = 2
        yc_dim = 49
        yt_dim = 49
    elif config.dset == "asos":
        x_dim = 6
        yc_dim = 6
        yt_dim = 6
    elif config.dset == "solar_energy":
        x_dim = 6
        yc_dim = 137
        yt_dim = 137
    elif config.dset == "exchange":
        x_dim = 6
        yc_dim = 8
        yt_dim = 8
    elif config.dset == "toy1":
        x_dim = 6
        yc_dim = 20
        yt_dim = 20
    elif config.dset == "toy2":
        x_dim = 6
        yc_dim = 20
        yt_dim = 20
    elif config.dset == "mnist":
        x_dim = 1
        yc_dim = 1
        yt_dim = 1
    elif config.dset == "cifar":
        x_dim = 1
        yc_dim = 3
        yt_dim = 3
    elif config.dset == "copy":
        x_dim = 1
        yc_dim = config.copy_vars
        yt_dim = config.copy_vars
    elif config.dset == "crypto":
        x_dim = 6
        yc_dim = 18
        yt_dim = 18
    elif config.dset == "toy_eeg":
        x_dim = 6
        yc_dim = 5
        yt_dim = 5
    elif config.dset == 'eeg_social_memory':
        x_dim = cf.read("x_dim")  #4  # mod 6
        yc_dim = num_channels
        yt_dim = num_channels
    assert x_dim is not None
    assert yc_dim is not None
    assert yt_dim is not None

    if config.model == "lstm":
        forecaster = stf.lstm_model.LSTM_Forecaster(
            # encoder
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            time_emb_dim=config.time_emb_dim,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            dropout_p=config.dropout_p,
            # training
            learning_rate=config.learning_rate,
            teacher_forcing_prob=config.teacher_forcing_start,
            l2_coeff=config.l2_coeff,
            loss=config.loss,
            linear_window=config.linear_window,
        )
    elif config.model == "mtgnn":
        forecaster = stf.mtgnn_model.MTGNN_Forecaster(
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            context_points=config.context_points,
            target_points=config.target_points,
            gcn_depth=config.gcn_depth,
            dropout_p=config.dropout_p,
            node_dim=config.node_dim,
            dilation_exponential=config.dilation_exponential,
            conv_channels=config.conv_channels,
            subgraph_size=config.subgraph_size,
            skip_channels=config.skip_channels,
            end_channels=config.end_channels,
            residual_channels=config.residual_channels,
            layers=config.layers,
            propalpha=config.propalpha,
            tanhalpha=config.tanhalpha,
            learning_rate=config.learning_rate,
            kernel_size=config.kernel_size,
            l2_coeff=config.l2_coeff,
            time_emb_dim=config.time_emb_dim,
            loss=config.loss,
            linear_window=config.linear_window,
        )
    elif config.model == "lstnet":
        forecaster = stf.lstnet_model.LSTNet_Forecaster(
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            context_points=config.context_points,
            hidRNN=config.hidRNN,
            hidCNN=config.hidCNN,
            hidSkip=config.hidSkip,
            CNN_kernel=config.CNN_kernel,
            skip=config.skip,
            dropout_p=config.dropout_p,
            output_fun=config.output_fun,
            learning_rate=config.learning_rate,
            l2_coeff=config.l2_coeff,
            loss=config.loss,
            linear_window=config.linear_window,
        )
    elif config.model == "spacetimeformer":
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
    elif config.model == "linear":
        forecaster = stf.linear_model.Linear_Forecaster(
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            context_points=config.context_points,
            learning_rate=config.learning_rate,
            l2_coeff=config.l2_coeff,
            loss=config.loss,
            linear_window=config.linear_window,
        )
    elif config.model == "s4":
        forecaster = stf.s4_model.S4_Forecaster(
            context_points=config.context_points,
            target_points=config.target_points,
            d_state=config.d_state,
            d_model=config.d_model,
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            layers=config.layers,
            time_emb_dim=config.time_emb_dim,
            channels=config.channels,
            dropout_p=config.dropout_p,
            learning_rate=config.learning_rate,
            l2_coeff=config.l2_coeff,
            loss=config.loss,
            linear_window=config.linear_window,
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
            RUN_NICKNAME = cf.read('run_nickname')
            SUBJECT_ID = cf.read('subject_id')
            CONDITION = cf.read('condition')
            #data_path = f"./data/eeg_social_memory/{RUN_NICKNAME}/ch{channel}_sub_{SUBJECT_ID}_cond_{CONDITION}.pkl"
            #data_path = f"./data/Generated EEG/eeg_generated_ch{channel}_sub_{cf.read('subject_id')}.pkl"
            data_path = f"./data/eeg_hyperscanning/{RUN_NICKNAME}/ch{channel}_sub_{SUBJECT_ID}_cond_{CONDITION}.pkl"
            print(data_path)

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
        NULL_VAL = None

    return DATA_MODULE, INV_SCALER, SCALER, NULL_VAL, PLOT_VAR_IDXS, PLOT_VAR_NAMES


def create_callbacks(config):
    saving = pl.callbacks.ModelCheckpoint(
        #dirpath=f"./data/stf_model_checkpoints/{config.run_name}_{''.join([str(random.randint(0,9)) for _ in range(9)])}",  # mod
        dirpath=f"./plots_checkpoints_logs/{config.run_name}/checkpoints/{config.run_id}",
        #monitor="train/norm_mse", #modDebug "val/mse"
        monitor="val/mse", #modDebug "val/mse"
        mode="min",
        filename=f"{config.run_name}_" + "{epoch:02d}-{val/norm_mse:.2f}", #mod val/mse
        save_top_k=1,
    )
    callbacks = [saving]

    '''
    callbacks.append(
        pl.callbacks.early_stopping.EarlyStopping(
            #monitor="train/norm_mse", #modDebug val/loss
            monitor="val/loss", #modDebug val/loss
            patience=8, #modDebug 5
        )
    )
    '''

    if config.wandb:
        callbacks.append(pl.callbacks.LearningRateMonitor())

    if config.model == "lstm":
        callbacks.append(
            stf.callbacks.TeacherForcingAnnealCallback(
                start=config.teacher_forcing_start,
                end=config.teacher_forcing_end,
                steps=config.teacher_forcing_anneal_steps,
            )
        )
    if config.time_mask_loss:
        callbacks.append(
            stf.callbacks.TimeMaskedLossCallback(
                start=config.time_mask_start,
                end=config.target_points,
                steps=config.time_mask_anneal_steps,
            )
        )
    return callbacks


def main(args):
    # add
    num_channels = int(cf.read('num_channels'))
    CHANNEL_START = 0
    CHANNEL_END = num_channels
    for channel in range(CHANNEL_START, CHANNEL_END+1):
        write_run_id(channel)
        if channel == CHANNEL_START:
            num_channels -= 1
            cf.write('num_channels', num_channels)
        if channel == CHANNEL_END:
            num_channels += 1
            cf.write('num_channels', num_channels)

        args.run_name, _, args.run_id = read_config_file()
        print(f"args.run_name: {args.run_name}")

        if args.wandb:
            import wandb
            #mod
            os.environ["PYTHONWARNINGS"] = "ignore:semaphore_tracker:UserWarning"
            STF_WANDB_ACCT="luca_maurici"
            STF_WANDB_PROJ="EEG connectivity estimate with transformers"
            # optionally: change wandb logging directory (defaults to ./data/STF_LOG_DIR)
            #STF_LOG_DIR="./data/STF_LOG_DIR"    
            #STF_LOG_DIR=f"./plots_checkpoints_logs/{args.run_name}/wandb_logs/{args.run_id}"  # mod and commented
            # mod
            project = STF_WANDB_PROJ
            entity = STF_WANDB_ACCT

            # commented
            '''
            log_dir = STF_LOG_DIR
            if log_dir is None:
                log_dir = "./data/STF_LOG_DIR"
                print(
                    "Using default wandb log dir path of ./data/STF_LOG_DIR. This can be adjusted with the environment variable `STF_LOG_DIR`"
                )

            print(f'\nproject {project}, entity {entity}')

            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            assert (
                project is not None and entity is not None
            ), "Please set environment variables `STF_WANDB_ACCT` and `STF_WANDB_PROJ` with \n\
                your wandb user/organization name and project title, respectively."
            '''

            experiment = wandb.init(
                project=project,
                entity=entity,
                config=args,
                #dir=log_dir,
                reinit=True,
            )
            config = wandb.config
            wandb.run.name = args.run_name
            wandb.run.save()
            logger = pl.loggers.WandbLogger(
                experiment=experiment#, save_dir="./data/stf_LOG_DIR"
            )
            logger.log_hyperparams(config)

        # Dset
        (
            data_module,
            inv_scaler,
            scaler,
            null_val,
            plot_var_idxs,
            plot_var_names,
        ) = create_dset(args, channel)

        # Model
        args.null_value = null_val
        #args.pad_value = pad_val  # new
        forecaster = create_model(args, num_channels)
        forecaster.set_inv_scaler(inv_scaler)
        forecaster.set_scaler(scaler)
        forecaster.set_null_value(null_val)

        # Callbacks
        callbacks = create_callbacks(args)
        test_samples, test_samples_unscaled = next(iter(data_module.train_dataloader()))

        if args.wandb and args.plot:
            callbacks.append(
                stf.plot.PredictionPlotterCallback(
                    test_samples,
                    var_idxs=plot_var_idxs,
                    var_names=plot_var_names,
                    total_samples=min(16, args.batch_size),
                )
            )

        if args.wandb and args.dset in ["mnist", "cifar"] and args.plot:
            callbacks.append(
                stf.plot.ImageCompletionCallback(
                    test_samples,
                    total_samples=min(16, args.batch_size),
                )
            )

        if args.wandb and args.dset == "copy" and args.plot:
            callbacks.append(
                stf.plot.CopyTaskCallback(
                    test_samples,
                    total_samples=min(16, args.batch_size),
                )
            )

        if args.wandb and args.model == "spacetimeformer" and args.attn_plot:
            callbacks.append(
                stf.plot.AttentionMatrixCallback(
                    test_samples,
                    layer=0,
                    total_samples=min(16, args.batch_size),  # mod 16
                )
            )

        trainer = pl.Trainer(
            gpus=args.gpus,
            callbacks=callbacks,
            logger=logger if args.wandb else None,
            accelerator="auto",
            gradient_clip_val=args.grad_clip_norm,
            gradient_clip_algorithm="norm",
            overfit_batches=1 if args.debug else 0,
            accumulate_grad_batches=args.accumulate,
            sync_batchnorm=True,
            val_check_interval=args.val_check_interval,
            limit_val_batches=args.limit_val_batches,
            max_epochs=45
        )

        # Train
        trainer.fit(forecaster, datamodule=data_module)

        # Test
        #trainer.test(datamodule=data_module, ckpt_path="best")
        '''
        # Predict (only here as a demo and test)
        forecaster.to("cuda")
        #xc, yc, xt, _ = test_samples  # mod
        xc, yc_us, xt, _ = test_samples_unscaled
        yc, scaler_data = scaler(yc_us)
        yc = torch.from_numpy(yc).float()
        yt_pred = forecaster.predict(xc, yc, xt)

        yt_pred_unscaled = inv_scaler(yt_pred, scaler_data)
        '''

        if args.wandb:
            experiment.finish()


if __name__ == "__main__":
    # CLI
    parser = create_parser()
    args = parser.parse_args()

    for trial in range(args.trials):
        main(args)
