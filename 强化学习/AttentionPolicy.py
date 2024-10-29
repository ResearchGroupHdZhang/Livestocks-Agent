import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution, make_proba_distribution
import torch


class CustomAttentionExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 128):
        super(CustomAttentionExtractor, self).__init__(observation_space, features_dim)

        self.attention_layer = nn.MultiheadAttention(embed_dim=features_dim, num_heads=4, batch_first=True)

        self.fc_amount_in = nn.Linear(observation_space['Amount_Move_in'].shape[1], features_dim)
        self.layernorm_amount_in = nn.LayerNorm(features_dim)
        self.fc_amount_out = nn.Linear(observation_space['Amount_Move_out'].shape[1], features_dim)
        self.layernorm_amount_out = nn.LayerNorm(features_dim)
        self.fc_n_demand_in = nn.Linear(observation_space['N_demand_Move_in'].shape[1], features_dim)
        self.layernorm_n_demand_in = nn.LayerNorm(features_dim)
        self.fc_n_demand_out = nn.Linear(observation_space['N_demand_Move_out'].shape[1], features_dim)
        self.layernorm_n_demand_out = nn.LayerNorm(features_dim)
        self.fc_ammonia_density_in = nn.Linear(observation_space['ammonia_density_Move_in'].shape[1], features_dim)
        self.layernorm_ammonia_density_in = nn.LayerNorm(features_dim)
        self.fc_ammonia_density_out = nn.Linear(observation_space['ammonia_density_Move_out'].shape[1], features_dim)
        self.layernorm_ammonia_density_out = nn.LayerNorm(features_dim)
        self.fc_livestock_pb_in = nn.Linear(observation_space['livestock_PB_Move_in'].shape[1], features_dim)
        self.layernorm_livestock_pb_in = nn.LayerNorm(features_dim)
        self.fc_livestock_pb_out = nn.Linear(observation_space['livestock_PB_Move_out'].shape[1], features_dim)
        self.layernorm_livestock_pb_out = nn.LayerNorm(features_dim)
        self.fc_relative_pm25_in = nn.Linear(observation_space['relative_pm25_Move_in'].shape[1], features_dim)
        self.layernorm_relative_pm25_in = nn.LayerNorm(features_dim)
        self.fc_relative_pm25_out = nn.Linear(observation_space['relative_pm25_Move_out'].shape[1], features_dim)
        self.layernorm_relative_pm25_out = nn.LayerNorm(features_dim)
        self.fc_sensitivity_in = nn.Linear(observation_space['sensitivity_Move_in'].shape[1], features_dim)
        self.layernorm_sensitivity_in = nn.LayerNorm(features_dim)
        self.fc_sensitivity_out = nn.Linear(observation_space['sensitivity_Move_out'].shape[1], features_dim)
        self.layernorm_sensitivity_out = nn.LayerNorm(features_dim)

        # Transformer block
        self.transformer_layer_in = nn.TransformerEncoderLayer(d_model=features_dim, nhead=4)
        self.transformer_layer_out = nn.TransformerEncoderLayer(d_model=features_dim, nhead=4)

        self.fusion_layer = nn.MultiheadAttention(embed_dim=features_dim, num_heads=4, batch_first=True)
        self.fc1 = nn.Linear(features_dim, features_dim)
        self.fc2 = nn.Linear(features_dim, features_dim)

        dummy_input = {
            "Amount_Move_in": th.zeros((1,) + observation_space['Amount_Move_in'].shape),
            "Amount_Move_out": th.zeros((1,) + observation_space['Amount_Move_out'].shape),
            "N_demand_Move_in": th.zeros((1,) + observation_space['N_demand_Move_in'].shape),
            "N_demand_Move_out": th.zeros((1,) + observation_space['N_demand_Move_out'].shape),
            "ammonia_density_Move_in": th.zeros((1,) + observation_space['ammonia_density_Move_in'].shape),
            "ammonia_density_Move_out": th.zeros((1,) + observation_space['ammonia_density_Move_out'].shape),
            "livestock_PB_Move_in": th.zeros((1,) + observation_space['livestock_PB_Move_in'].shape),
            "livestock_PB_Move_out": th.zeros((1,) + observation_space['livestock_PB_Move_out'].shape),
            "relative_pm25_Move_in": th.zeros((1,) + observation_space['relative_pm25_Move_in'].shape),
            "relative_pm25_Move_out": th.zeros((1,) + observation_space['relative_pm25_Move_out'].shape),
            "sensitivity_Move_in": th.zeros((1,) + observation_space['sensitivity_Move_in'].shape),
            "sensitivity_Move_out": th.zeros((1,) + observation_space['sensitivity_Move_out'].shape),
        }
        self.forward(dummy_input)

    def forward(self, observations):
        amount_in_tensor = observations["Amount_Move_in"]
        amount_out_tensor = observations["Amount_Move_out"]
        n_demand_in_tensor = observations["N_demand_Move_in"]
        n_demand_out_tensor = observations["N_demand_Move_out"]
        ammonia_density_in_tensor = observations["ammonia_density_Move_in"]
        ammonia_density_out_tensor = observations["ammonia_density_Move_out"]
        livestock_pb_in_tensor = observations["livestock_PB_Move_in"]
        livestock_pb_out_tensor = observations["livestock_PB_Move_out"]
        relative_pm25_in_tensor = observations["relative_pm25_Move_in"]
        relative_pm25_out_tensor = observations["relative_pm25_Move_out"]
        sensitivity_in_tensor = observations["sensitivity_Move_in"]
        sensitivity_out_tensor = observations["sensitivity_Move_out"]

        # 通过全连接层处理各个特征

        x1 = th.relu(self.fc_amount_in(amount_in_tensor))
        x1 = self.layernorm_amount_in(x1)
        x2 = th.relu(self.fc_amount_out(amount_out_tensor))
        x2 = self.layernorm_amount_out(x2)
        x3 = th.relu(self.fc_n_demand_in(n_demand_in_tensor))
        x3 = self.layernorm_n_demand_in(x3)
        x4 = th.relu(self.fc_n_demand_out(n_demand_out_tensor))
        x4 = self.layernorm_n_demand_out(x4)
        x5 = th.relu(self.fc_ammonia_density_in(ammonia_density_in_tensor))
        x5 = self.layernorm_ammonia_density_in(x5)
        x6 = th.relu(self.fc_ammonia_density_out(ammonia_density_out_tensor))
        x6 = self.layernorm_ammonia_density_out(x6)
        x7 = th.relu(self.fc_livestock_pb_in(livestock_pb_in_tensor))
        x7 = self.layernorm_livestock_pb_in(x7)
        x8 = th.relu(self.fc_livestock_pb_out(livestock_pb_out_tensor))
        x8 = self.layernorm_livestock_pb_out(x8)
        x9 = th.relu(self.fc_relative_pm25_in(relative_pm25_in_tensor))
        x9 = self.layernorm_relative_pm25_in(x9)
        x10 = th.relu(self.fc_relative_pm25_out(relative_pm25_out_tensor))
        x10 = self.layernorm_relative_pm25_out(x10)
        x11 = th.relu(self.fc_sensitivity_in(sensitivity_in_tensor))
        x11 = self.layernorm_sensitivity_in(x11)
        x12 = th.relu(self.fc_sensitivity_out(sensitivity_out_tensor))
        x12 = self.layernorm_sensitivity_out(x12)

        obs_tensor_in = th.cat([x1, x3, x5, x7, x9, x11], dim=1)
        obs_tensor_out = th.cat([x2, x4, x6, x8, x10, x12], dim=1)


        x_in = th.relu(self.fc1(obs_tensor_in))
        x_out = th.relu(self.fc2(obs_tensor_out))

        # Transformer block
        x_in = self.transformer_layer_in(x_in)
        x_out = self.transformer_layer_out(x_out)

        # fusion block
        x, _ = self.fusion_layer(x_in, x_out, x_out)

        x = th.mean(x, dim=1)
        return x

class CustomAttentionPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomAttentionPolicy, self).__init__(*args, **kwargs, 
                                                    features_extractor_class=CustomAttentionExtractor,
                                                    features_extractor_kwargs=dict(features_dim=128))

        # 改变输出动作的分布，如果是多离散动作空间需要根据具体动作分布修改
        self.action_dist = make_proba_distribution(self.action_space)
    # def _get_action_dist_from_latent(self, latent_pi: th.Tensor, mask=None):
    #     """
    #     Retrieve action distribution given the latent codes.

    #     :param latent_pi: Latent code for the actor
    #     :return: Action distribution
    #     """
    #     mean_actions = self.action_net(th.mean(latent_pi,dim=1))

    #     # Here mean_actions are the logits before the softmax
    #     if mask is not None:
    #         return self.action_dist.proba_distribution(action_logits=mean_actions, mask=mask)
    #     else:
    #         return self.action_dist.proba_distribution(action_logits=mean_actions)
