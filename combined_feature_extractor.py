from modified_efficient_net import ModifiedEfficientNet, get_modified_efficientnet
from modified_swin_arch import ModifiedSwinTransformer, get_modified_swintransformer
import torch
from torch import nn

class CombinedFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.modified_eff_net = get_modified_efficientnet()
        self.modified_swin = get_modified_swintransformer()

        # Use the out_channels attributes directly
        combined_feature_dim = self.modified_eff_net.out_channels + self.modified_swin.out_channels

        # Combined feature layer and new classification head
        self.combined_layer = nn.Sequential(
            nn.Linear(combined_feature_dim, combined_feature_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.classification_head = nn.Linear(combined_feature_dim, 10)

    def forward(self, x):
        eff_features = self.modified_eff_net(x)
        swin_features = self.modified_swin(x)
        combined_features = torch.cat([eff_features, swin_features], dim=1)
        combined_features = self.combined_layer(combined_features)
        # Remove the classification head
        # combined_features = self.classification_head(combined_features)
        return combined_features