from torch_geometric.graphgym.register import register_config


@register_config("overwrite_defaults")
def overwrite_defaults_cfg(cfg):
    """Overwrite core GraphGym defaults for the ST-PINN mainline."""

    cfg.train.mode = "custom"
    cfg.dataset.name = "none"
    cfg.round = 5


@register_config("extended_cfg")
def extended_cfg(cfg):
    """General extended config options for the active topology model."""

    cfg.name_tag = ""
    cfg.train.ckpt_best = False

    # Old diffusion_backbone_only-style supervision:
    # split retained edges and newly added edges, then weight them separately.
    cfg.model.lambda_old = 1.0
    cfg.model.lambda_new_start = 1.0
    cfg.model.lambda_new_final = 1.0
    cfg.model.lambda_new_warmup_epochs = 50
    cfg.model.lambda_new_schedule = "linear"

    # Optional conservation regularizer for the simplified ST-PINN.
    # `lambda_con` is the final target value when staged scheduling is used.
    cfg.model.lambda_con = 0.05
    cfg.model.lambda_con_schedule = "staged_linear"
    cfg.model.lambda_con_zero_epochs = 50
    cfg.model.lambda_con_mid = 0.01
    cfg.model.lambda_con_mid_epoch = 120
    cfg.model.lambda_con_final_epoch = 200
    cfg.model.lambda_con_warmup_epochs = 50

    cfg.train.current_epoch = 0
    cfg.train.log_variance_stats = False
    cfg.train.eval_test_during_training = False

    cfg.optim.wsd_stable_epochs = 0
    cfg.optim.wsd_decay_epochs = 50
    cfg.optim.wsd_decay_type = "cosine"
