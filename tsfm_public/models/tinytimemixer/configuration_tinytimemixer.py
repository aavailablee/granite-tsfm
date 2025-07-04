# Copyright contributors to the TSFM project
#
"""TinyTimeMixer model configuration"""

from typing import Optional, Union

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

TINYTIMEMIXER_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class TinyTimeMixerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`TinyTimeMixerModel`]. It is used to instantiate a
    TinyTimeMixer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the TinyTimeMixer {} architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        context_length (`int`, *optional*, defaults to 64)
            The context/history length for the input sequence.
        patch_length (`int`, *optional*, defaults to 8)
            The patch length for the input sequence.
        num_input_channels (`int`):
            Number of input variates. For Univariate, set it to 1.
        patch_stride (`int`, *optional*, defaults to 8):
            Amount of points to stride. If its value is same as patch_length, we get non-overlapping patches.
        d_model (`int`, *optional*, defaults to 16):
            Hidden feature size of the model.
        prediction_length (`int`, *optional*, defaults to 16)
            Number of time steps to forecast for a forecasting task. Also known as the Forecast Horizon.
        num_parallel_samples (`int`, *optional*, defaults to 100):
            The number of samples to generate in parallel for probabilistic forecast.
        expansion_factor (`int`, *optional*, defaults to 2):
            Expansion factor to use inside MLP. Recommended range is 2-5. Larger value indicates more complex model.
        num_layers (`int`, *optional*, defaults to 3):
            Number of layers to use. Recommended range is 3-15. Larger value indicates more complex model.
        dropout (`float`, *optional*, defaults to 0.2):
            The dropout probability the `TinyTimeMixer` backbone. Recommended range is 0.2-0.7
        mode (`str`, *optional*, defaults to `"common_channel"`):
            Mixer Mode. Determines how to process the channels. Allowed values: "common_channel", "mix_channel". In
            "common_channel" mode, we follow Channel-independent modelling with no explicit channel-mixing. Channel
            mixing happens in an implicit manner via shared weights across channels. (preferred first approach) In
            "mix_channel" mode, we follow explicit channel-mixing in addition to patch and feature mixer. (preferred
            approach when channel correlations are very important to model)
        gated_attn (`bool`, *optional*, defaults to `True`):
            Enable Gated Attention.
        norm_mlp (`str`, *optional*, defaults to `"LayerNorm"`):
            Normalization layer (BatchNorm or LayerNorm).
        self_attn (`bool`, *optional*, defaults to `False`):
            Enable Tiny self attention across patches. This can be enabled when the output of Vanilla TinyTimeMixer with
            gated attention is not satisfactory. Enabling this leads to explicit pair-wise attention and modelling
            across patches.
        self_attn_heads (`int`, *optional*, defaults to 1):
            Number of self-attention heads. Works only when `self_attn` is set to `True`.
        use_positional_encoding (`bool`, *optional*, defaults to `False`):
            Enable the use of positional embedding for the tiny self-attention layers. Works only when `self_attn` is
            set to `True`.
        positional_encoding_type (`str`, *optional*, defaults to `"sincos"`):
            Positional encodings. Options `"random"` and `"sincos"` are supported. Works only when
            `use_positional_encoding` is set to `True`
        scaling (`string` or `bool`, *optional*, defaults to `"std"`):
            Whether to scale the input targets via "mean" scaler, "std" scaler or no scaler if `None`. If `True`, the
            scaler is set to "mean".
        loss (`string`, *optional*, defaults to `"mse"`):
            The loss function to finetune or pretrain the the model. Allowed values are "mse" or "mae" or "pinball" or "huber".
            Use pinball loss for probabilistic forecasts of different quantiles.
            Distribution head (nll) is currently disabled and not allowed.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated normal weight initialization distribution.
        post_init (`bool`, *optional*, defaults to `False`):
            Whether to use custom weight initialization from `transformers` library, or the default initialization in
            `PyTorch`. Setting it to `False` performs `PyTorch` weight initialization.
        norm_eps (`float`, *optional*, defaults to 1e-05):
            A value added to the denominator for numerical stability of normalization.
        adaptive_patching_levels (`int`, *optional*, defaults to 0):
            If adaptive_patching_levels is i, then we will have i levels with each level having n_layers.
            Level id starts with 0. num_patches at level i will be multipled by (2^i) and num_features at level i will be divided by (2^i).
            For Ex. if adaptive_patching_levels is 3 - then we will have 3 levels:
                level 2: num_features//(2^2), num_patches*(2^2)
                level 1: num_features//(2^1), num_patches*(2^1)
                level 0: num_features//(2^0), num_patches*(2^0)
            adaptive_patching_levels = 1 is same as one level PatchTSMixer. This module gets disabled when adaptive_patching_levels is 0 or neg value. Defaults to 0 (off mode).
        resolution_prefix_tuning (`bool`, *optional*, defaults to `False`):
            Enable if your dataloader has time resolution information as defined in `get_freq_mapping` function in `modelling_tinytimemixer`.
        frequency_token_vocab_size (`int`, *optional*, defaults to 5):
            Vocab size to use when resolution_prefix_tuning is enabled.
        head_dropout (`float`, *optional*, defaults to 0.2):
            The dropout probability the `TinyTimeMixer` head.
        distribution_output (`string`, *optional*, defaults to `"student_t"`):
            The distribution emission head for the model when loss is "nll". Could be either "student_t", "normal" or
            "negative_binomial".
        prediction_channel_indices (`list`, *optional*):
            List of channel indices to forecast. If None, forecast all channels. Target data is expected to have all
            channels and we explicitly filter the channels in prediction and target before loss computation. Please provide the indices
            in sorted ascending order.
        exogenous_channel_indices (`list`, *optional*):
            List of channel indices whose values are known in the forecast period. Please provide the indices
            in sorted ascending order.
        decoder_num_layers (`int`, *optional*, defaults to 8):
            Number of layers to use in decoder
        decoder_d_model(`int`, *optional*, defaults to 16):
            Defines the hidden feature size of the decoder.
        decoder_adaptive_patching_levels (`int`, *optional*, defaults to 0):
            Adaptive Patching levels for decoder. Preferable to set it to 0 for decoder to keep it light weight.
        decoder_raw_residual (`bool`, *optional*, defaults to `False`):
            Flag to enable merging of raw embedding with encoder embedding for decoder input. Defaults to False.
        decoder_mode (`string`, *optional*, defaults to `"common_channel"`):
            Decoder channel mode. Use `"common_channel" for channel-independent modelling and `"mix_channel"` for channel-mixing modelling
        use_decoder (`bool`, *optional*, defaults to `True`):
            Enable to use decoder.
        enable_forecast_channel_mixing (`bool`, *optional*, defaults to `False`):
            Enable if we want to reconcile forecasts across all channels and also to enable exogenous infusion, if you have them.
        fcm_gated_attn (`bool`, *optional*, defaults to `True`):
            Enable gated attention in forecast channel mixing block.
        fcm_context_length (`int`, *optional*, defaults to `1):
            Surrounding context length to use. For Ex. If we want to consider 2 lag point before and after a data point, provide value 2 for `fcm_context_length`
        fcm_use_mixer (`bool`, *optional*, defaults to `True`):
            Enable Mixing in forecast channel mixing block.
        fcm_mix_layers (`int`, *optional*, defaults to 2):
            Number of mixer layers to use if fcm_use_mixer is enabled
        fcm_prepend_past (`bool`, *optional*, defaults to `True`):
            Prepend last context for forecast reconciliation
        fcm_prepend_past_offset  (`int`, *optional*, defaults to None):

        categorical_vocab_size_list (`list`, *optional*):
            List of vocab size for all the tokenized categorical variables to use. Pass it in the same order as used in the foreward call param `static_categorical_values`.
        prediction_filter_length (`int`,*optional*, defaults to None):
            Actual length in the prediction output to use for loss calculations.


    Example:

    ```python
    >>> from transformers import TinyTimeMixerConfig, TinyTimeMixerModel

    >>> # Initializing a default TinyTimeMixer configuration
    >>> configuration = TinyTimeMixerConfig()

    >>> # Randomly initializing a model (with random weights) from the configuration
    >>> model = TinyTimeMixerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "tinytimemixer"
    attribute_map = {
        "hidden_size": "d_model",
        "num_hidden_layers": "num_layers",
    }

    def __init__(
        self,
        # Time series specific configuration
        context_length: int = 64,
        patch_length: int = 8,
        num_input_channels: int = 1,
        prediction_length: int = 16,
        patch_stride: int = 8,
        prediction_channel_indices: Optional[list] = None,
        exogenous_channel_indices: Optional[list] = None,
        # General model configuration
        d_model: int = 16,
        expansion_factor: int = 2,
        num_layers: int = 3,
        dropout: float = 0.2,
        mode: str = "common_channel",
        gated_attn: bool = True,
        norm_mlp: str = "LayerNorm",
        self_attn: bool = False,
        self_attn_heads: int = 1,
        use_positional_encoding: bool = False,
        positional_encoding_type: str = "sincos",
        scaling: Optional[Union[str, bool]] = "std",
        loss: Optional[str] = "mse",
        init_std: float = 0.02,
        post_init: bool = False,
        norm_eps: float = 1e-5,
        adaptive_patching_levels: int = 0,
        resolution_prefix_tuning: bool = False,
        frequency_token_vocab_size: int = 5,
        # General head configuration
        head_dropout: float = 0.2,
        distribution_output: str = "student_t",
        num_parallel_samples: int = 100,
        # decoder parameters
        decoder_num_layers: int = 8,
        decoder_d_model: int = 8,
        decoder_adaptive_patching_levels: int = 0,
        decoder_raw_residual: bool = False,
        decoder_mode: str = "common_channel",
        use_decoder: bool = True,
        # forecast channel mixing wit exog support
        enable_forecast_channel_mixing: bool = False,
        fcm_gated_attn: bool = True,
        fcm_context_length: int = 1,
        fcm_use_mixer: bool = False,
        fcm_mix_layers: int = 2,
        fcm_prepend_past: bool = True,
        fcm_prepend_past_offset: Optional[int] = None,
        # static categorical
        categorical_vocab_size_list: Optional[list] = None,
        # prediction length filtering
        prediction_filter_length: Optional[int] = None,
        # initialization parameters
        init_linear: str = "pytorch",
        init_embed: str = "pytorch",
        quantile: float = 0.5,
        huber_delta: float = 1,
        momentum: list = [0.9, 0.99, 0.999],
        enc_in: int = 7,
        bsa: bool = True,
        batch_size: int = 64,
        **kwargs,
    ):
        self.num_input_channels = num_input_channels
        self.context_length = context_length
        self.patch_length = patch_length
        self.expansion_factor = expansion_factor
        self.num_layers = num_layers
        self.dropout = dropout
        self.mode = mode
        self.gated_attn = gated_attn
        self.norm_mlp = norm_mlp
        self.scaling = scaling
        self.head_dropout = head_dropout

        self.patch_last = True
        self.use_positional_encoding = use_positional_encoding
        self.positional_encoding_type = positional_encoding_type
        self.prediction_length = prediction_length
        self.prediction_channel_indices = prediction_channel_indices
        self.self_attn = self_attn
        self.self_attn_heads = self_attn_heads
        self.init_std = init_std
        self.post_init = post_init
        self.distribution_output = distribution_output
        self.loss = loss
        self.num_parallel_samples = num_parallel_samples
        self.norm_eps = norm_eps

        self.use_decoder = use_decoder

        self.adaptive_patching_levels = adaptive_patching_levels
        self.resolution_prefix_tuning = resolution_prefix_tuning
        self.exogenous_channel_indices = exogenous_channel_indices
        self.decoder_num_layers = decoder_num_layers
        self.decoder_adaptive_patching_levels = decoder_adaptive_patching_levels
        self.decoder_raw_residual = decoder_raw_residual
        self.decoder_mode = decoder_mode
        self.fcm_gated_attn = fcm_gated_attn
        self.fcm_context_length = fcm_context_length
        self.fcm_use_mixer = fcm_use_mixer
        self.fcm_mix_layers = fcm_mix_layers
        self.fcm_prepend_past = fcm_prepend_past
        self.fcm_prepend_past_offset = fcm_prepend_past_offset
        self.enable_forecast_channel_mixing = enable_forecast_channel_mixing
        self.frequency_token_vocab_size = frequency_token_vocab_size

        self.d_model = d_model
        self.patch_stride = patch_stride
        self.decoder_d_model = decoder_d_model
        self.categorical_vocab_size_list = categorical_vocab_size_list
        self.init_processing = False
        self.prediction_filter_length = prediction_filter_length
        self.init_linear = init_linear
        self.init_embed = init_embed
        self.quantile = quantile
        self.huber_delta = huber_delta

        self.momentum_params = momentum
        self.batch_size = batch_size
        self.bptt = True
        self.enc_in = int(enc_in)
        self.bsa = bsa

        super().__init__(**kwargs)

    def check_and_init_preprocessing(self):
        self.init_processing = True

        if not hasattr(self, "num_patches"):
            context_length = self.context_length
            self.num_patches = (max(context_length, self.patch_length) - self.patch_length) // self.patch_stride + 1

            if self.resolution_prefix_tuning:
                self.num_patches += 1

        if self.prediction_filter_length is not None:
            if self.prediction_filter_length > self.prediction_length or self.prediction_filter_length <= 0:
                raise ValueError("prediction_filter_length should be positive and less than prediction_length")

        if self.loss == "nll" and self.enable_forecast_channel_mixing:
            raise ValueError("Distribution head cannot be enabled when enable_forecast_channel_mixing is set to True")

        if self.prediction_channel_indices is not None:
            self.prediction_channel_indices.sort()

        if self.exogenous_channel_indices is not None:
            self.exogenous_channel_indices.sort()

        if self.exogenous_channel_indices is not None and self.prediction_channel_indices is None:
            self.prediction_channel_indices = list(
                set(range(self.num_input_channels)) - set(self.exogenous_channel_indices)
            )
            self.prediction_channel_indices.sort()
