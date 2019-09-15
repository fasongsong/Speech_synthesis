import tensorflow as tf

ref_dim = 256
def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = tf.contrib.training.HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=5000,
        iters_per_checkpoint=300,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=False,
        distributed_run=True,

        dist_backend="nccl",
        dist_url="tcp://localhost:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,

        ################################
        # Data Parameters             #
        ################################
        load_mel_from_disk=True,
        training_files='filelists/metadata_mel10_train.csv',
        validation_files='filelists/metadata_mel10_val.csv',
        text_cleaners=['korean_cleaners'], # english_cleaners, korean_cleaners
        sort_by_length=False,
        mel_time_warping=True,
        mel_freq_warping=True,
        mel_time_length_adjustment=False,
        mel_time_length_adjustment_double=False,
        mel_time_mask=False,
        mel_freq_mask=False,
        value_adjustmet=False,

        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        sampling_rate=22050,
        filter_length=1024,
        hop_length=256, # number audio of frames between stft colmns, default win_length/4
        win_length=1024, # win_length int <= n_ftt: fft window size (frequency domain), defaults to win_length = n_fft
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,
        max_abs_mel_value = 4.0,

        ################################
        # Model Parameters             #
        ################################
        n_symbols = 80, # set 80 if u use korean_cleaners. set 149 if u use english_cleaners
        symbols_embedding_dim=512,
        speaker_num = 10,
        # # Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,
        n_convolutions=6,
        conv_dim_in=[1, 32, 32, 64, 64, 128],
        conv_dim_out=[32, 32, 64, 64, 128, ref_dim],
        # source mel prenet parameters
        mel_fc_dim = 256,
        mel_rnn_dim = 256,

        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        decoder_rnn_dim=256,
        prenet_dim=256,
        max_decoder_steps=1000,
        gate_threshold=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,

        # Attention parameters
        attention_rnn_dim=256,
        attention_dim=128,

        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31,

        # Mel-post processing network parameters
        postnet_embedding_dim=256,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=1e-3,
        weight_decay=1e-6,
        grad_clip_thresh=1.0,
        batch_size=32,
        mask_padding=True,  # set model's padded outputs to padded values

        ################################
        # VAE Hyperparameters #
        ################################
        anneal_function='logistic',
        anneal_k=0.0025,
        anneal_x0=3000,
        anneal_upper=0.2,
        anneal_lag=50000,

        reference_dim=256,
        z_speaker_dim=64,
        z_residual_dim=8,

        # speaker_list=["fv01", "fv02", "fv03", "fv04", "fv05", "fv06", "fv07", "fv08", "fv09", "fv10", "fv11", "fv12",
        #               "fv13", "fv14", "fv15", "fv16", "fv17", "fv18", "fv19", "fv20", "fx01", "fx02", "fx03", "fx04",
        #               "fx05", "fx06", "fx07", "fx08", "fx09", "fx10", "fx11", "fx12", "fx13", "fx14", "fx15", "fx16",
        #               "fx17", "fx18", "fx19", "fx20", "fy01", "fy02", "fy03", "fy04", "fy05", "fy06", "fy07", "fy08",
        #               "fy09", "fy10", "fy11", "fy12", "fy13", "fy14", "fy16", "fy17", "fy18", "fz05", "fz06",
        #               "mv01", "mv02", "mv03", "mv04", "mv05", "mv06", "mv07", "mv08", "mv09", "mv10", "mv11", "mv12",
        #               "mv13", "mv14", "mv15", "mv16", "mv17", "mv18", "mv19", "mv20", "mw01", "mw02", "mw03", "mw04",
        #               "mw05", "mw06", "mw07", "mw08", "mw09", "mw10", "mw11", "mw13", "mw14", "mw15", "mw16",
        #               "mw17", "mw18", "mw19", "mw20", "my01", "my02", "my03", "my04", "my05", "my06", "my07", "my08",
        #               "my09", "my10", "my11", "mz01", "mz02", "mz03", "mz04", "mz05", "mz06", "mz07", "mz08", "mz09"]
        speaker_list=["fv01", "fv02", "fv03", "fv04", "fv05", "mv01", "mv02", "mv03", "mv04", "mv05"]

    )

    if hparams_string:
        tf.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams
