from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import robomimic.utils.tensor_utils as TensorUtils
from egomimic.algo import register_algo_factory_func, PolicyAlgo
from robomimic.algo.bc import BC

from egomimic.utils.egomimicUtils import nds, DynamicWrapper
import matplotlib.pyplot as plt
import robomimic.utils.obs_utils as ObsUtils

from egomimic.configs import config_factory

from robomimic.models.base_nets import Vit
import egomimic.models.policy_nets as PolicyNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.models.transformers import PositionalEncoding

from egomimic.models.hpt_nets import *

from egomimic.utils.hpt_utils import *

import json

import hydra
from omegaconf import OmegaConf

from functools import partial
from typing import List, Optional
import numpy as np
import einops
from collections import defaultdict

#TODO: write comments

class HPTModel(nn.Module):
    """
    Heterogenous Pretrained Transformer implementation from the HPT paper but with some additional modifications


    """

    def __init__(
        self,
        embed_dim=1024,
        num_blocks=24,
        num_heads=16,
        token_postprocessing="action_token",
        observation_horizon=4,
        action_horizon=1,
        no_trunk=False,
        shared_modality_trunk=None,
        use_domain_embedding=False,
        drop_path=0.0,
        weight_init_style="pytorch",
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.shared_modality_trunk = shared_modality_trunk
        self.no_trunk = no_trunk

        self.encoders = nn.ModuleDict()

        self.trunk = self._create_policy_trunk(
            embed_dim=embed_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            drop_path=drop_path,
            weight_init_style=weight_init_style,
        )

        self.stems = {}
        self.heads = {}
        # self.normalizer = {}
        self.encoders = {}
        self.domains = []
        self.use_modality_embedding = use_modality_embedding
        self.observation_horizon = observation_horizon
        self.action_horizon = action_horizon
        self.token_postprocessing = token_postprocessing
        self.modalities_tokens = {}
        self.action_tokens = None
        self.stem_spec = {}
        self.head_spec = {}

        self.modalities = {}

        self.shared_keys = []

        self.auxiliary_key = None

    def init_encoder(self, modality, encoder_spec):
        """
        """
        encoder = hydra.utils.instantiate(encoder_spec)
        self.encoders[modality] = encoder

    def init_domain_stem(self, domain_name, stem_spec):
        """
        """
        self.stem_spec[domain_name] = stem_spec
        self.modalities[domain_name] = stem_spec.keys()

        for modality in self.modalities[domain_name]:
            stem_name = f"{domain_name}_{modality}"
            self.stems[stem_name] = hydra.utils.instantiate(getattr(stem_spec, modality))
            if hasattr(self.stems[stem_name], 'init_cross_attn'):
                self.stems[stem_name].init_cross_attn(stem_spec[modality].specs.cross_attn_specs, modality)

            self.modalities_tokens[modality] = nn.Parameter(
                torch.randn(1, 1, stem_spec[modality].specs.cross_attn_specs.modality_embed_dim) * STD_SCALE
            )
        
    def init_domain_head(self, domain_name, head_spec):
        """
        """
        self.head_spec[domain_name] = head_spec
        self.domains.append(domain_name)
        self.heads[domain_name] = hydra.utils.instantiate(head_spec)
    
    def finalize_modules(self):
        self.stems = nn.ModuleDict(self.stems)
        self.heads = nn.ModuleDict(self.heads)
        self.modalities_tokens = nn.ParameterDict(self.modalities_tokens)
        self.apply(self._init_weights)

        ## Shared action tokens
        if self.token_postprocessing == "action_token":
            self.action_tokens = nn.Parameter(
                torch.randn(1, self.action_horizon, self.embed_dim) * STD_SCALE
            )
    
    def _create_policy_trunk(self, embed_dim, num_blocks, num_heads, drop_path, weight_init_style):
        """
        #TODO: Make this hydra instantiate
        """
        trunk = {}
        
        trunk["trunk"] = SimpleTransformer(
            embed_dim=embed_dim,
            num_blocks=num_blocks,
            ffn_dropout_rate=0.0,
            drop_path_rate=drop_path,
            attn_target=partial(
                MultiheadAttention,
                embed_dim=embed_dim,
                num_heads=num_heads,
                bias=True,
                add_bias_kv=True,
            ),
            pre_transformer_layer=nn.Sequential(
                nn.Identity(),
                EinOpsRearrange("b l d -> l b d"),
            ),
            post_transformer_layer=EinOpsRearrange("l b d -> b l d"),
            weight_init_style=weight_init_style,
        )
        if hasattr(self, "shared_modality_trunk") and self.shared_modality_trunk is not None:
            for modality in self.shared_modality_trunk.modalities:
                trunk[modality] = self.shared_modality_trunk[modality]

        return nn.ModuleDict(trunk)

    def get_position_embedding(self, feature, embed_dim):
        """
        """
        tokensize = int(feature.shape[1])
        tokens = get_sinusoid_encoding_table(0, tokensize, self.embed_dim)
        return tokens.repeat((1, 1, 1)).to(feature.device)
    
    def preprocess_tokens(self, domain, features):
        """
        """
        tokens = torch.cat(features, dim=-2)
        
        if self.token_postprocessing == "action_token":
            action_tokens = self.action_tokens.repeat(len(tokens), 1, 1)
            tokens = torch.cat([tokens, action_tokens], dim=-2)

        position_tokens = self.get_position_embedding(tokens, self.embed_dim)
        return tokens + position_tokens
    
    def postprocess_tokens(self, trunk_tokens):
        """
        """
        if self.token_postprocessing == "mean":
            return trunk_tokens.mean(dim=1)
        elif self.token_postprocessing == "action_token":
            return trunk_tokens[:, -self.action_horizon:]
        elif self.token_postprocessing == "max":
            return trunk_tokens.max(dim=1)[0]
        elif self.token_postprocessing == "last":
            return trunk_tokens[:, -1]
        elif self.token_postprocessing == "no-op":
            return trunk_tokens
        else:
            raise ValueError(f"Invalid token_postprocessing: {self.token_postprocessing}")

    def preprocess_states(self, domain, data):
        """
        """
        if "state" in data:
            data["state"] = data["state"][:, :, None]
        if f"state_{self.auxiliary_key}" in data:
            data[f"state_{self.auxiliary_key}"] = data[f"state_{self.auxiliary_key}"][:, :, None]
        return data

    def stem_process(self, domain, data):
        feats = []
        feat_dict = {}
        for modality in (list(self.modalities[domain]) + self.shared_keys):
            if modality not in data:
                continue
            
            if modality in self.shared_keys:
                domain = "shared"
            
            stem = self.stems[f"{domain}_{modality}"]
            if modality in self.encoders:
                data[modality] = self.encoders[modality](data[modality])

            data_shape = data[modality].shape
            data_horizon = data_shape[1]
            horizon = data_horizon

            if getattr(self, "train_mode", False) and self.stem_spec[domain][modality].cross_attn_specs.random_horizon_masking and data_horizon > 1:
                horizon = np.random.randint(1, data_horizon + 1)
                data[modality] = data[modality][:, data_horizon - horizon:]
            
            positional_embedding = get_sinusoid_encoding_table(
                0, horizon * int(np.prod(data_shape[2:-1])), data_shape[-1]
            ).to(data[modality])
            positional_embedding = einops.repeat(
                positional_embedding, 
                "b h w -> (repeat b) h w", 
                repeat=data_shape[0]
            )

            data[modality] = data[modality] + positional_embedding.view(data[modality].shape)
            stem_token = stem.compute_latent(data[modality])
            feats.append(stem_token)
            feat_dict[modality] = stem_token

        return feats, feat_dict

    def get_visual_embeds(self, domain, data, modality):
        """
        """
        if modality in self.shared_keys:
            domain = "shared"

        stem = self.stems[f"{domain}_{modality}"]

        encoder_feats = None

        if modality in self.encoders:
            encoder_feats = self.encoders[modality](data[modality])
        data_shape = encoder_feats.shape
        data_horizon = data_shape[1]
        horizon = data_horizon

        positional_embedding = get_sinusoid_encoding_table(
                0, horizon * int(np.prod(data_shape[2:-1])), data_shape[-1]
            ).to(encoder_feats)
        positional_embedding = einops.repeat(
                positional_embedding, 
                "b h w -> (repeat b) h w", 
                repeat=data_shape[0]
            )
        stem_feats = encoder_feats + positional_embedding.view(encoder_feats.shape)
        stem_token = stem.compute_latent(stem_feats)
        return [encoder_feats, stem_token]
    
    def forward_features(self, domain, data):
        """
        """
        data = self.preprocess_states(domain, data)
        stem_tokens, token_dict = self.stem_process(domain, data)
        if self.early_fusion:
            stem_tokens = self.early_fusion_process(domain, token_dict)
        trunk_tokens = self.preprocess_tokens(domain, stem_tokens)

        if not self.no_trunk:
            trunk_tokens = self.trunk["trunk"](trunk_tokens)

        return self.postprocess_tokens(trunk_tokens)
    
    def compute_loss(self, batch):
        """
        """
        self.train_mode = True
        domain, data = batch["domain"][0], batch["data"]
        
        features = self.forward_features(domain, data)

        if f"state_{self.auxiliary_key}" in data:
            domain = f"{domain}_{self.auxiliary_key}" 

        return self.heads[domain].compute_loss(features, data)
    
    def forward(self, domain, data):
        """ 
        """
        features = self.forward_features(domain, data)
        
        if f"state_{self.auxiliary_key}" in data:
            domain = f"{domain}_{self.auxiliary_key}"

        action = self.heads[domain](features)
        return action 

    def save(self, checkpoint_path="./.checkpoints/hpt/full/"):
        """
        """
        try:
            torch.save(self.state_dict(), checkpoint_path)
        except FileNotFoundError:
            print(f"Could not save module parameters for trunk to {checkpoint_path}.")
    
    def _init_weights(self, m):
        """
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def freeze_trunk(self, num_layers=0):
        """
        """
        layers = list(self.trunk["trunk"].children())
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = False
    
    def unfreeze_trunk(self, num_layers=0):
        """
        """
        layers = list(self.trunk["trunk"].children())
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

    def load_trunk(self, path):
        """
        """
        if "hf://" in path:
            if "output" in path:
                path = path.replace("output/", "")
            path = download_from_huggingface(path[len("hf://") :])
        self.trunk.load_state_dict(torch.load(path), strict=True)

    def load_pretrained(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            checkpoint_path = download_from_huggingface(checkpoint_path[len("hf://") :])
        
        self.load_trunk(os.path.join(checkpoint_path, "trunk.pth"))

class HPT(Algo):
    """
    """
    def __init__(
        self,
        data_schematic,
        camera_transforms,
        # ---------------------------
        # Trunk params
        # ---------------------------
        trunk: dict,
        # ---------------------------
        # Other model params
        # ---------------------------
        stem_specs: dict = None,
        head_specs: dict = None,
        shared_stem_specs: dict = None,
        shared_obs_keys: list = None,
        encoder_specs: dict = None,
        domains: list = None,
        auxiliary_domains: list = None,
        auxiliary_key: str = None,
        # ---------------------------
        # Pretrained
        # ---------------------------
        pretrained: bool = False,
        pretrained_checkpoint: str = "",
        # ---------------------------
        # Optional image augmentations
        # ---------------------------
        train_image_augs=None,
        eval_image_augs=None,
        # ---------------------------
        # Catch-all kwargs
        # ---------------------------
        **kwargs
    ):
        self.nets = nn.ModuleDict()
        self.data_schematic = data_schematic

        self.camera_transforms = camera_transforms
        self.stem_specs = stem_specs
        self.head_specs = head_specs
        self.encoders = encoder_specs

        self.shared_stem_specs = shared_stem_specs
        self.shared_obs_keys = shared_obs_keys

        self.pretrained = pretrained
        self.pretrained_checkpoint = pretrained_checkpoint
        
        self.domains = domains.copy()
        self.auxiliary_domains = auxiliary_domains.copy()
        self.auxiliary_key = self.auxiliary_key

        model = HPTModel(**trunk)
        model.auxiliary_key = self.auxiliary_key

        self.camera_keys = data_schematic.keys_of_type("camera_keys")
        self.proprio_keys = data_schematic.keys_of_type("proprio_keys")
        self.lang_keys = data_schematic.keys_of_type("lang_keys")

        self.proprio_keys = [key for key in self.proprio_keys if key not in self.lang_keys]
        self.obs_keys = self.proprio_keys + self.camera_keys + self.lang_keys

        self.multitask = kwargs.get(multitask, False)
        self.device = kwargs.get(device, torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        if self.pretrained:
            model.load_pretrained(self.pretrained_checkpoint)

        for domain in self.domains:
            model.init_domain_stem(domain, self.stem_specs[domain])
            model.init_domain_head(domain, self.head_specs[domain])
        
        if self.shared_obs_keys is not None:
            model.init_domain_stem("shared", self.shared_stem_specs)
            model.shared_keys = self.shared_obs_keys

        for domain in self.auxiliary_domains:
            model.init_domain_head(domain, self.head_specs[domain])
        
        for modality, encoder_cfg in self.encoders.items():
            model.init_encoder(modality, encoder_cfg)
        
        model.finalize_modules()

        self.ac_key_robot = data_schematic.keys_of_type("action_keys_robot")
        self.ac_key_hand = data_schematic.keys_of_type("action_keys_hand")

        self.nets["policy"] = model
        self.nets = self.nets.float().to(self.device)

    @override
    def process_batch_for_training(self, batch):
        batch = self.data_schematic.normalize_data(batch, self.embodiment_id)
        


    