def get_autoencoder_makers_and_configs(autoencoder_cfg, variance_type, img_shape):
    latent_channels = autoencoder_cfg["latent_channels"]

    encoder_configuration = {
        **autoencoder_cfg["encoder"],
        "img_resolution": img_shape[-1],
        "img_channels": img_shape[0],
        "out_channels": latent_channels * 2,  # output number of channels
    }
    decoder_configuration = {
        **autoencoder_cfg["decoder"],
        "img_resolution": img_shape[-1],
        "embedding_channels": latent_channels,
        "img_channels": img_shape[0] if variance_type != "diag" else 2 * img_shape[0],
    }
    if autoencoder_cfg["type"] == "edm":
        from latent_vae.networks import Encoder, Decoder
    elif autoencoder_cfg["type"] == "ldm":
        from latent_vae.networks_ldm import Encoder, Decoder
    else:
        raise NotImplementedError("Only edm and ldm encoder types are implemented")
    return Encoder, Decoder, encoder_configuration, decoder_configuration


def get_datasource(cfg):
    if cfg["name"] == "GaussDataModule":
        from datasources.gauss_spde import GaussVAEDataModule
        return GaussVAEDataModule(
            data_dir=cfg["data_dir"],
            batch_size=cfg["batch_size"],
            n_procs=cfg["n_procs"],
            val_ptg=cfg["val_ptg"]
            ), (1, 256, 256)
    else:
        raise NotImplementedError("Only GaussDataModule implemented")
