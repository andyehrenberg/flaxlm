from typing import Dict, Tuple
import os
from pickle import UnpicklingError

import msgpack.exceptions
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import unfreeze
from flax.serialization import from_bytes
from flax.traverse_util import flatten_dict, unflatten_dict
import transformers
from transformers import PretrainedConfig
from transformers.utils.hub import get_checkpoint_shard_files
from transformers.utils import (
    cached_file,
    download_url,
    has_file,
    is_remote_url,
    logging,
)

logger = logging.get_logger(__name__)

WEIGHTS_NAME = "pytorch_model.bin"
WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
FLAX_WEIGHTS_NAME = "flax_model.msgpack"
FLAX_WEIGHTS_INDEX_NAME = "flax_model.msgpack.index.json"


class LogicallyPartitionedModel(transformers.FlaxPreTrainedModel):
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        dtype: jnp.dtype = jnp.float32,
        *model_args,
        **kwargs,
    ):
        config = kwargs.pop("config", None)
        cache_dir = kwargs.pop("cache_dir", None)
        from_pt = kwargs.pop("from_pt", False)
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        _do_init = kwargs.pop("_do_init", True)
        subfolder = kwargs.pop("subfolder", "")
        commit_hash = kwargs.pop("_commit_hash", None)

        if trust_remote_code is True:
            logger.warning(
                "The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is"
                " ignored."
            )

        user_agent = {
            "file_type": "model",
            "framework": "flax",
            "from_auto_class": from_auto_class,
        }
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = (
                config if config is not None else pretrained_model_name_or_path
            )
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                subfolder=subfolder,
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline,
                _commit_hash=commit_hash,
                **kwargs,
            )
        else:
            model_kwargs = kwargs

        if commit_hash is None:
            commit_hash = getattr(config, "_commit_hash", None)

        # Add the dtype to model_kwargs
        model_kwargs["dtype"] = dtype

        # This variable will flag if we're loading a sharded checkpoint. In this case the archive file is just the
        # index of the files.
        is_sharded = False

        # Load model
        if pretrained_model_name_or_path is not None:
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)
            is_local = os.path.isdir(pretrained_model_name_or_path)
            if os.path.isdir(pretrained_model_name_or_path):
                if from_pt and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder, WEIGHTS_NAME)
                ):
                    # Load from a PyTorch checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, subfolder, WEIGHTS_NAME
                    )
                elif from_pt and os.path.isfile(
                    os.path.join(
                        pretrained_model_name_or_path, subfolder, WEIGHTS_INDEX_NAME
                    )
                ):
                    # Load from a sharded pytorch checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, subfolder, WEIGHTS_INDEX_NAME
                    )
                    is_sharded = True
                elif os.path.isfile(
                    os.path.join(
                        pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME
                    )
                ):
                    # Load from a Flax checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME
                    )
                elif os.path.isfile(
                    os.path.join(
                        pretrained_model_name_or_path,
                        subfolder,
                        FLAX_WEIGHTS_INDEX_NAME,
                    )
                ):
                    # Load from a sharded Flax checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path,
                        subfolder,
                        FLAX_WEIGHTS_INDEX_NAME,
                    )
                    is_sharded = True
                # At this stage we don't have a weight file so we will raise an error.
                elif os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder, WEIGHTS_NAME)
                ):
                    raise EnvironmentError(
                        f"Error no file named {FLAX_WEIGHTS_NAME} found in directory {pretrained_model_name_or_path} "
                        "but there is a file for PyTorch weights. Use `from_pt=True` to load this model from those "
                        "weights."
                    )
                else:
                    raise EnvironmentError(
                        f"Error no file named {FLAX_WEIGHTS_NAME} or {WEIGHTS_NAME} found in directory "
                        f"{pretrained_model_name_or_path}."
                    )
            elif os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path)):
                archive_file = pretrained_model_name_or_path
                is_local = True
            elif is_remote_url(pretrained_model_name_or_path):
                filename = pretrained_model_name_or_path
                resolved_archive_file = download_url(pretrained_model_name_or_path)
            else:
                filename = WEIGHTS_NAME if from_pt else FLAX_WEIGHTS_NAME
                try:
                    # Load from URL or cache if already cached
                    cached_file_kwargs = dict(
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        resume_download=resume_download,
                        local_files_only=local_files_only,
                        use_auth_token=use_auth_token,
                        user_agent=user_agent,
                        revision=revision,
                        subfolder=subfolder,
                        _raise_exceptions_for_missing_entries=False,
                        _commit_hash=commit_hash,
                    )
                    resolved_archive_file = cached_file(
                        pretrained_model_name_or_path, filename, **cached_file_kwargs
                    )

                    # Since we set _raise_exceptions_for_missing_entries=False, we don't get an expection but a None
                    # result when internet is up, the repo and revision exist, but the file does not.
                    if resolved_archive_file is None and filename == FLAX_WEIGHTS_NAME:
                        # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path,
                            FLAX_WEIGHTS_INDEX_NAME,
                            **cached_file_kwargs,
                        )
                        if resolved_archive_file is not None:
                            is_sharded = True
                    # Maybe the checkpoint is pytorch sharded, we try to grab the pytorch index name in this case.
                    elif resolved_archive_file is None and from_pt:
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path,
                            WEIGHTS_INDEX_NAME,
                            **cached_file_kwargs,
                        )
                        if resolved_archive_file is not None:
                            is_sharded = True
                    if resolved_archive_file is None:
                        # Otherwise, maybe there is a TF or Flax model file.  We try those to give a helpful error
                        # message.
                        has_file_kwargs = {
                            "revision": revision,
                            "proxies": proxies,
                            "use_auth_token": use_auth_token,
                        }
                        if has_file(
                            pretrained_model_name_or_path,
                            WEIGHTS_NAME,
                            **has_file_kwargs,
                        ):
                            raise EnvironmentError(
                                f"{pretrained_model_name_or_path} does not appear to have a file named"
                                f" {FLAX_WEIGHTS_NAME} but there is a file for PyTorch weights. Use `from_pt=True` to"
                                " load this model from those weights."
                            )
                        elif has_file(
                            pretrained_model_name_or_path,
                            WEIGHTS_INDEX_NAME,
                            **has_file_kwargs,
                        ):
                            raise EnvironmentError(
                                f"{pretrained_model_name_or_path} does not appear to have a file named"
                                f" {FLAX_WEIGHTS_INDEX_NAME} but there is a sharded file for PyTorch weights. Use"
                                " `from_pt=True` to load this model from those weights."
                            )
                        else:
                            raise EnvironmentError(
                                f"{pretrained_model_name_or_path} does not appear to have a file named"
                                f" {FLAX_WEIGHTS_NAME} or {WEIGHTS_NAME}."
                            )
                except EnvironmentError:
                    # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted
                    # to the original exception.
                    raise
                except Exception:
                    # For any other exception, we throw a generic error.
                    raise EnvironmentError(
                        f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it"
                        " from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                        f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                        f" directory containing a file named {FLAX_WEIGHTS_NAME} or {WEIGHTS_NAME}."
                    )

            if is_local:
                logger.info(f"loading weights file {archive_file}")
                resolved_archive_file = archive_file
            else:
                logger.info(
                    f"loading weights file {filename} from cache at {resolved_archive_file}"
                )
        else:
            resolved_archive_file = None

        # We'll need to download and cache each checkpoint shard if the checkpoint is sharded.
        if is_sharded:
            # resolved_archive_file becomes a list of files that point to the different checkpoint shards in this case.
            resolved_archive_file, _ = get_checkpoint_shard_files(
                pretrained_model_name_or_path,
                resolved_archive_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                user_agent=user_agent,
                revision=revision,
                subfolder=subfolder,
                _commit_hash=commit_hash,
            )

        # init random models
        model = cls(config, *model_args, _do_init=_do_init, **model_kwargs)

        if from_pt:
            state = load_pytorch_checkpoint_in_flax_state_dict(
                model, resolved_archive_file, is_sharded
            )
        else:
            if is_sharded:
                state = cls.load_flax_sharded_weights(resolved_archive_file)
            else:
                try:
                    with open(resolved_archive_file, "rb") as state_f:
                        state = from_bytes(cls, state_f.read())
                except (UnpicklingError, msgpack.exceptions.ExtraData) as e:
                    try:
                        with open(resolved_archive_file) as f:
                            if f.read().startswith("version"):
                                raise OSError(
                                    "You seem to have cloned a repository without having git-lfs installed. Please"
                                    " install git-lfs and run `git lfs install` followed by `git lfs pull` in the"
                                    " folder you cloned."
                                )
                            else:
                                raise ValueError from e
                    except (UnicodeDecodeError, ValueError):
                        raise EnvironmentError(
                            f"Unable to convert {archive_file} to Flax deserializable object. "
                        )
            # make sure all arrays are stored as jnp.arrays
            # NOTE: This is to prevent a bug this will be fixed in Flax >= v0.3.4:
            # https://github.com/google/flax/issues/1261
            if _do_init:
                state = jax.tree_util.tree_map(jnp.array, state)
            else:
                # keep the params on CPU if we don't want to initialize
                state = jax.tree_util.tree_map(
                    lambda x: jax.device_put(x, jax.devices("cpu")[0]), state
                )

        # if model is base model only use model_prefix key
        if (
            cls.base_model_prefix not in dict(model.params_shape_tree)
            and cls.base_model_prefix in state
        ):
            state = state[cls.base_model_prefix]

        # if model is head model and we are loading weights from base model
        # we initialize new params dict with base_model_prefix
        if (
            cls.base_model_prefix in dict(model.params_shape_tree)
            and cls.base_model_prefix not in state
        ):
            state = {cls.base_model_prefix: state}

        # flatten dicts
        state = flatten_dict(state)

        random_state = flatten_dict(
            unfreeze(model.params if _do_init else model.params_shape_tree)
        )

        missing_keys = model.required_params - set(state.keys())
        unexpected_keys = set(state.keys()) - model.required_params

        if missing_keys and not _do_init:
            logger.warning(
                f"The checkpoint {pretrained_model_name_or_path} is missing required keys: {missing_keys}. "
                "Make sure to call model.init_weights to initialize the missing weights."
            )
            cls._missing_keys = missing_keys

        # Mistmatched keys contains tuples key/shape1/shape2 of weights in the checkpoint that have a shape not
        # matching the weights in the model.
        mismatched_keys = []
        for key in state.keys():
            state_key = state[key].value if hasattr(state[key], "value") else state[key]
            random_state_key = (
                random_state[key].value
                if hasattr(random_state[key], "value")
                else random_state[key]
            )
            if key in random_state and state_key.shape != random_state_key.shape:
                if ignore_mismatched_sizes:
                    mismatched_keys.append(
                        (key, state_key.shape, random_state[key].shape)
                    )
                    state_key = random_state_key
                else:
                    raise ValueError(
                        f"Trying to load the pretrained weight for {key} failed: checkpoint has shape "
                        f"{state_key.shape} which is incompatible with the model shape {random_state_key.shape}. "
                        "Using `ignore_mismatched_sizes=True` if you really want to load this checkpoint inside this "
                        "model."
                    )
            state[key] = nn.LogicallyPartitioned(
                value=state[key], names=random_state[key].names
            )

        # add missing keys as random parameters if we are initializing
        if missing_keys and _do_init:
            for missing_key in missing_keys:
                state[missing_key] = random_state[missing_key]

        # remove unexpected keys to not be saved again
        for unexpected_key in unexpected_keys:
            del state[unexpected_key]

        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when"
                f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
                f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or"
                " with another architecture (e.g. initializing a BertForSequenceClassification model from a"
                " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
                f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical"
                " (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
            )
        else:
            logger.info(
                f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n"
            )

        if len(missing_keys) > 0:
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably"
                " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        elif len(mismatched_keys) == 0:
            logger.info(
                f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the checkpoint"
                f" was trained on, you can already use {model.__class__.__name__} for predictions without further"
                " training."
            )
        if len(mismatched_keys) > 0:
            mismatched_warning = "\n".join(
                [
                    f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                    for key, shape1, shape2 in mismatched_keys
                ]
            )
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized because the shapes did not"
                f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able"
                " to use it for predictions and inference."
            )

        # dictionary of key: dtypes for the model params
        param_dtypes = jax.tree_util.tree_map(lambda x: x.dtype, state)
        # extract keys of parameters not in jnp.float32
        fp16_params = [k for k in param_dtypes if param_dtypes[k] == jnp.float16]
        bf16_params = [k for k in param_dtypes if param_dtypes[k] == jnp.bfloat16]

        # raise a warning if any of the parameters are not in jnp.float32
        if len(fp16_params) > 0:
            logger.warning(
                f"Some of the weights of {model.__class__.__name__} were initialized in float16 precision from "
                f"the model checkpoint at {pretrained_model_name_or_path}:\n{fp16_params}\n"
                "You should probably UPCAST the model weights to float32 if this was not intended. "
                "See [`~FlaxPreTrainedModel.to_fp32`] for further information on how to do this."
            )

        if len(bf16_params) > 0:
            logger.warning(
                f"Some of the weights of {model.__class__.__name__} were initialized in bfloat16 precision from "
                f"the model checkpoint at {pretrained_model_name_or_path}:\n{bf16_params}\n"
                "You should probably UPCAST the model weights to float32 if this was not intended. "
                "See [`~FlaxPreTrainedModel.to_fp32`] for further information on how to do this."
            )

        if _do_init:
            # set correct parameters
            model.params = unflatten_dict(state)
            return model
        else:
            return model, unflatten_dict(state)


def load_pytorch_checkpoint_in_flax_state_dict(
    flax_model, pytorch_checkpoint_path, is_sharded, allow_missing_keys=False
):
    """Load pytorch checkpoints in a flax model"""
    try:
        import torch  # noqa: F401
    except ImportError:
        logger.error(
            "Loading a PyTorch model in Flax, requires both PyTorch and Flax to be installed. Please see"
            " https://pytorch.org/ and https://flax.readthedocs.io/en/latest/installation.html for installation"
            " instructions."
        )
        raise

    if not is_sharded:
        pt_path = os.path.abspath(pytorch_checkpoint_path)
        logger.info(f"Loading PyTorch weights from {pt_path}")

        pt_state_dict = torch.load(pt_path, map_location="cpu")
        logger.info(
            f"PyTorch checkpoint contains {sum(t.numel() for t in pt_state_dict.values()):,} parameters."
        )

        flax_state_dict = convert_pytorch_state_dict_to_flax(pt_state_dict, flax_model)
    else:
        # model is sharded and pytorch_checkpoint_path already contains the list of .pt shard files
        flax_state_dict = convert_pytorch_sharded_state_dict_to_flax(
            pytorch_checkpoint_path, flax_model
        )
    return flax_state_dict


def rename_key_and_reshape_tensor(
    pt_tuple_key: Tuple[str],
    pt_tensor: np.ndarray,
    random_flax_state_dict: Dict[str, jnp.ndarray],
    model_prefix: str,
) -> Tuple[Tuple[str], np.ndarray]:
    """Rename PT weight names to corresponding Flax weight names and reshape tensor if necessary"""

    def is_key_or_prefix_key_in_dict(key: Tuple[str]) -> bool:
        """Checks if `key` of `(prefix,) + key` is in random_flax_state_dict"""
        return len(set(random_flax_state_dict) & set([key, (model_prefix,) + key])) > 0

    # layer norm
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("scale",)
    if pt_tuple_key[-1] in ["weight", "gamma"] and is_key_or_prefix_key_in_dict(
        renamed_pt_tuple_key
    ):
        return renamed_pt_tuple_key, pt_tensor

    # embedding
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("embedding",)
    if pt_tuple_key[-1] == "weight" and is_key_or_prefix_key_in_dict(
        renamed_pt_tuple_key
    ):
        return renamed_pt_tuple_key, pt_tensor

    # conv layer
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("kernel",)
    if (
        pt_tuple_key[-1] == "weight"
        and pt_tensor.ndim == 4
        and not is_key_or_prefix_key_in_dict(pt_tuple_key)
    ):
        pt_tensor = pt_tensor.transpose(2, 3, 1, 0)
        return renamed_pt_tuple_key, pt_tensor

    # linear layer
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("kernel",)
    if pt_tuple_key[-1] == "weight" and not is_key_or_prefix_key_in_dict(pt_tuple_key):
        pt_tensor = pt_tensor.T
        return renamed_pt_tuple_key, pt_tensor

    # old PyTorch layer norm weight
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("weight",)
    if pt_tuple_key[-1] == "gamma":
        return renamed_pt_tuple_key, pt_tensor

    # old PyTorch layer norm bias
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("bias",)
    if pt_tuple_key[-1] == "beta":
        return renamed_pt_tuple_key, pt_tensor

    return pt_tuple_key, pt_tensor


def convert_pytorch_state_dict_to_flax(pt_state_dict, flax_model):
    # convert pytorch tensor to numpy
    pt_state_dict = {k: v.numpy() for k, v in pt_state_dict.items()}

    model_prefix = flax_model.base_model_prefix
    random_flax_state_dict = flatten_dict(flax_model.params)
    flax_state_dict = {}

    load_model_with_head_into_base_model = (model_prefix not in flax_model.params) and (
        model_prefix in set([k.split(".")[0] for k in pt_state_dict.keys()])
    )
    load_base_model_into_model_with_head = (model_prefix in flax_model.params) and (
        model_prefix not in set([k.split(".")[0] for k in pt_state_dict.keys()])
    )

    # Need to change some parameters name to match Flax names
    for pt_key, pt_tensor in pt_state_dict.items():

        pt_tuple_key = tuple(pt_key.split("."))

        # remove base model prefix if necessary
        has_base_model_prefix = pt_tuple_key[0] == model_prefix
        if load_model_with_head_into_base_model and has_base_model_prefix:
            pt_tuple_key = pt_tuple_key[1:]

        # Correctly rename weight parameters
        flax_key, flax_tensor = rename_key_and_reshape_tensor(
            pt_tuple_key, pt_tensor, random_flax_state_dict, model_prefix
        )

        # add model prefix if necessary
        require_base_model_prefix = (model_prefix,) + flax_key in random_flax_state_dict
        if load_base_model_into_model_with_head and require_base_model_prefix:
            flax_key = (model_prefix,) + flax_key

        if flax_key in random_flax_state_dict:
            arr = random_flax_state_dict[flax_key]
            if hasattr(arr, "value"):
                arr = arr.value
            if flax_tensor.shape != arr.shape:
                raise ValueError(
                    f"PyTorch checkpoint seems to be incorrect. Weight {pt_key} was expected to be of shape "
                    f"{arr.shape}, but is {flax_tensor.shape}."
                )

        # also add unexpected weight so that warning is thrown
        if hasattr(random_flax_state_dict[flax_key], "value"):
            flax_state_dict[flax_key] = nn.LogicallyPartitioned(
                value=jnp.asarray(flax_tensor),
                names=random_flax_state_dict[flax_key].names,
            )
        else:
            flax_state_dict[flax_key] = jnp.asarray(flax_tensor)

    return unflatten_dict(flax_state_dict)


def convert_pytorch_sharded_state_dict_to_flax(shard_filenames, flax_model):
    import torch

    # Load the index
    flax_state_dict = {}
    for shard_file in shard_filenames:
        # load using msgpack utils
        pt_state_dict = torch.load(shard_file)
        pt_state_dict = {k: v.numpy() for k, v in pt_state_dict.items()}

        model_prefix = flax_model.base_model_prefix
        random_flax_state_dict = flatten_dict(flax_model.params)

        load_model_with_head_into_base_model = (
            model_prefix not in flax_model.params
        ) and (model_prefix in set([k.split(".")[0] for k in pt_state_dict.keys()]))
        load_base_model_into_model_with_head = (model_prefix in flax_model.params) and (
            model_prefix not in set([k.split(".")[0] for k in pt_state_dict.keys()])
        )
        # Need to change some parameters name to match Flax names
        for pt_key, pt_tensor in pt_state_dict.items():

            pt_tuple_key = tuple(pt_key.split("."))

            # remove base model prefix if necessary
            has_base_model_prefix = pt_tuple_key[0] == model_prefix
            if load_model_with_head_into_base_model and has_base_model_prefix:
                pt_tuple_key = pt_tuple_key[1:]

            # Correctly rename weight parameters
            flax_key, flax_tensor = rename_key_and_reshape_tensor(
                pt_tuple_key, pt_tensor, random_flax_state_dict, model_prefix
            )
            # add model prefix if necessary
            require_base_model_prefix = (
                model_prefix,
            ) + flax_key in random_flax_state_dict
            if load_base_model_into_model_with_head and require_base_model_prefix:
                flax_key = (model_prefix,) + flax_key

            if flax_key in random_flax_state_dict:
                if flax_tensor.shape != random_flax_state_dict[flax_key].shape:
                    raise ValueError(
                        f"PyTorch checkpoint seems to be incorrect. Weight {pt_key} was expected to be of shape "
                        f"{random_flax_state_dict[flax_key].shape}, but is {flax_tensor.shape}."
                    )

            # also add unexpected weight so that warning is thrown
            flax_state_dict[flax_key] = jnp.asarray(flax_tensor)
    return unflatten_dict(flax_state_dict)
