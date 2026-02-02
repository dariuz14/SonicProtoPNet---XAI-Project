import torch
import torch.nn as nn
import torch.nn.functional as F

from receptive_field import compute_proto_layer_rf_info_v2
from helpers import conv_info_from_features
from backbone_extractor import get_backbone


class ProtoPNet(nn.Module):
    
    def __init__(self, backbone_name, prototype_shape, num_classes, img_size, add_on_layers_type='bottleneck', init_weights=True):
        super(ProtoPNet, self).__init__()

        self.backbone = get_backbone(backbone_name)
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.num_classes = num_classes
        self.epsilon = 1e-4

        assert self.num_prototypes % self.num_classes == 0 # Each class should have the same number of prototypes
        num_prototypes_per_class = self.num_prototypes // self.num_classes

        # One-hot encoding of prototype classes
        self.prototype_class_identity = torch.zeros(self.num_prototypes, self.num_classes)
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1

        layer_filter_sizes, layer_strides, layer_paddings = conv_info_from_features(self.backbone)
        self.proto_layer_rf_info = compute_proto_layer_rf_info_v2(img_size, layer_filter_sizes, layer_strides, layer_paddings, prototype_shape[2]) 

        # backbone_name = str(self.backbone).upper()
        backbone_name = backbone_name.upper()
        if backbone_name.startswith('VGG'):
            first_add_on_layer_in_channels = [i for i in self.backbone.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        elif backbone_name.startswith('DENSE'):
            first_add_on_layer_in_channels = [i for i in self.backbone.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
        else:
            raise Exception('Backbone not implemented') 

        if add_on_layers_type == 'bottleneck':
            add_on_layers = []
            current_in_channels = first_add_on_layer_in_channels
            while (current_in_channels > self.prototype_shape[1]) or (len(add_on_layers) == 0):
                current_out_channels = max(self.prototype_shape[1], current_in_channels // 2)
                add_on_layers.append(nn.Conv2d(in_channels=current_in_channels, out_channels=current_out_channels, kernel_size=1))
                add_on_layers.append(nn.ReLU())
                add_on_layers.append(nn.Conv2d(in_channels=current_out_channels, out_channels=current_out_channels, kernel_size=1))
                if current_out_channels > self.prototype_shape[1]:
                    add_on_layers.append(nn.ReLU())
                else:
                    assert current_out_channels == self.prototype_shape[1]
                    add_on_layers.append(nn.Sigmoid())
                current_in_channels = current_out_channels
            self.add_on_layers = nn.Sequential(*add_on_layers)
        else:
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1], kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
                nn.Sigmoid()
            )
        
        self.prototype_layer = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)

        self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)

        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes)

        if init_weights:
            self._initialize_weights()
    
    def _calculate_l2_distances(self, features):
        features_squared = features ** 2
        features_squared_sum = F.conv2d(input=features_squared, weight=self.ones)

        prototypes_squared = self.prototype_layer ** 2
        prototypes_squared = torch.sum(prototypes_squared, dim=(1,2,3))

        pt_reshaped = prototypes_squared.view(-1, 1, 1)

        features_p = F.conv2d(input=features, weight=self.prototype_layer)
        intermediate_result = -2 * features_p + pt_reshaped

        distances = F.relu(features_squared_sum + intermediate_result)

        return distances
    
    def _distances_to_similarity(self, distances):
        return torch.log((distances + 1) / (distances + self.epsilon))

    def forward(self, x):
        # f
        features = self.backbone(x)
        features = self.add_on_layers(features) 

        # g
        distances = self._calculate_l2_distances(features)

        min_distances = - F.max_pool2d(-distances, kernel_size=(distances.size()[2], distances.size()[3]))
        min_distances = min_distances.view(-1, self.num_prototypes)
        prototype_activations = self._distances_to_similarity(min_distances)

        # h
        logits = self.last_layer(prototype_activations)

        return logits, min_distances
    
    def push_forward(self, x):
        features = self.backbone(x)
        features = self.add_on_layers(features) 
        distances = self._calculate_l2_distances(features)
        return features, distances

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)
        
    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)
