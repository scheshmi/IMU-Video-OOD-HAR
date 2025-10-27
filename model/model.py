import torch
from torch import nn
import torch.nn.functional as F
from transformers import VideoMAEConfig, VideoMAEModel, PatchTSTConfig, PatchTSTModel
import numpy as np
import clip


class IMUPTST(nn.Module):
    def __init__(self, emd_dim, num_classes, model_name="namctin/patchtst_etth1_pretrain", pretrained=True, trainable=True):
        super().__init__()
        config = PatchTSTConfig(use_cls_token=True, num_input_channels=6,
                                context_length=250, patch_length=16, patch_stride=16)
        if pretrained:
            self.model = PatchTSTModel.from_pretrained(
                model_name, use_cls_token=True, num_input_channels=6)
        else:
            self.model = PatchTSTModel(config)

        for p in self.model.parameters():
            p.requires_grad = trainable

        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(emd_dim, num_classes)

    def forward(self, x):
        output = self.model(x)
        cls_token = output.last_hidden_state[:, :, 0, :]
        cls_token = self.flatten(cls_token)
        logits = self.classifier(cls_token)
        return logits


class LinearProbing(nn.Module):
    def __init__(self, clip_model, emd_dim, num_classes, trainable=False, joint_emd=True):
        super().__init__()
        self.clip_model = clip_model
        self.joint_emd = joint_emd

        for p in self.clip_model.parameters():
            p.requires_grad = trainable

        self.classifier_joint = nn.Linear(emd_dim, num_classes)
        self.classifier = nn.Linear(emd_dim, num_classes)

    def forward(self, x):
        x = self.clip_model.imu_encoder(x)

        if self.joint_emd:
            x = self.clip_model.imu_projection(x)
            output = self.classifier_joint(x)
        else:
            output = self.classifier(x)

        return output


class IMUEncoder(nn.Module):
    def __init__(self, model_name="namctin/patchtst_etth1_pretrain", pretrained=True, trainable=True):
        super().__init__()
        config = PatchTSTConfig(use_cls_token=True, num_input_channels=6,
                                context_length=250, patch_length=16, patch_stride=16)
        if pretrained:
            self.model = PatchTSTModel.from_pretrained(
                model_name, use_cls_token=True, num_input_channels=6)
        else:
            self.model = PatchTSTModel(config)

        for p in self.model.parameters():
            p.requires_grad = trainable

        self.flatten = nn.Flatten()

    def forward(self, x):
        output = self.model(x)
        cls_token = output.last_hidden_state[:, :, 0, :]
        return self.flatten(cls_token)


class VideoEncoder(nn.Module):
    def __init__(self, model_name="MCG-NJU/videomae-base-ssv2", clip_length=10, pretrained=True, trainable=True):
        super().__init__()
        if pretrained:
            self.model = VideoMAEModel.from_pretrained(model_name, num_frames=clip_length)
        else:
            config = VideoMAEConfig(num_frames=clip_length)
            self.model = VideoMAEModel(config)

        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        output = self.model(x)
        last_hidden_state = output.last_hidden_state
        return torch.mean(last_hidden_state, dim=1)


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim=512, dropout=0.1):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class IMUVideoCrossModel(nn.Module):
    def __init__(self, temperature=1.0, video_embedding=768, imu_embedding=6*128, bias=-10, loss_type='sigmoid'):
        super().__init__()
        self.video_encoder = VideoEncoder()
        self.imu_encoder = IMUEncoder(pretrained=False)
        self.video_projection = ProjectionHead(embedding_dim=768)
        self.imu_projection = ProjectionHead(embedding_dim=imu_embedding)
        self.temperature = nn.Parameter(torch.ones([]) * temperature)
        self.bias = nn.Parameter(torch.ones([]) * bias)
        self.loss_type = loss_type

    def cross_entropy(self, preds, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()

    def forward(self, batch):
        video_features = self.video_encoder(batch["video"])
        imu_features = self.imu_encoder(batch["imu"])

        video_embeddings = self.video_projection(video_features)
        imu_embeddings = self.imu_projection(imu_features)

        video_embeddings = F.normalize(video_embeddings, p=2, dim=-1)
        imu_embeddings = F.normalize(imu_embeddings, p=2, dim=-1)

        if self.loss_type == 'softmax':
            logits = (imu_embeddings @ video_embeddings.T) / self.temperature
            video_similarity = video_embeddings @ video_embeddings.T
            imu_similarity = imu_embeddings @ imu_embeddings.T
            targets = F.softmax(
                (video_similarity + imu_similarity) / 2 * self.temperature, dim=-1
            )
            imu_loss = self.cross_entropy(logits, targets, reduction='none')
            video_loss = self.cross_entropy(logits.T, targets.T, reduction='none')
            loss = (video_loss + imu_loss) / 2.0
            return loss.mean()
        else:
            logits = (imu_embeddings @ video_embeddings.T) * self.temperature.exp() + self.bias
            n = logits.size(0)
            labels = 2 * torch.eye(n, device=logits.device) - 1
            return -torch.sum(F.logsigmoid(labels * logits)) / n





class IMU2CLIP(nn.Module):
    def __init__(self, freeze=True, video_encoder_name="clip_1frame", temperature=0.1):
        super(IMU2CLIP, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.imu_encoder = MW2StackRNNPooling(size_embeddings=512)
        self.imu_projection = ProjectionHead(embedding_dim=512, projection_dim=768)
        self.video_projection = ProjectionHead(embedding_dim=512, projection_dim=768)
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.flag_freeze = freeze
        self.video_encoder_name = video_encoder_name

        if self.flag_freeze:
            self.clip_model.eval()

    def forward(self, batch):
        imu_embeddings = self.imu_encoder(batch["imu"].permute(0, 2, 1))
        video_embeddings = self.get_video_embeddings(batch["video"])
        video_embeddings = self.video_projection(video_embeddings)
        imu_embeddings = self.imu_projection(imu_embeddings)

        video_embeddings = F.normalize(video_embeddings, p=2, dim=-1)
        imu_embeddings = F.normalize(imu_embeddings, p=2, dim=-1)

        logits = (video_embeddings @ imu_embeddings.T) / self.temperature
        video_similarity = video_embeddings @ video_embeddings.T
        imu_similarity = imu_embeddings @ imu_embeddings.T
        targets = F.softmax(
            (video_similarity + imu_similarity) / 2 * self.temperature, dim=-1
        )
        imu_loss = self.cross_entropy(logits, targets, reduction='none')
        video_loss = self.cross_entropy(logits.T, targets.T, reduction='none')
        loss = (video_loss + imu_loss) / 2.0
        return loss.mean()

    def cross_entropy(self, preds, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()

    def get_img_embeddings(self, img):
        img = img.float()
        if img.shape[-1] != 224 or img.shape[-2] != 224:
            img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
        img = img.type(self.clip_model.dtype)
        img_features = self.clip_model.encode_image(img)
        return img_features

    def get_video_embeddings(self, video):
        if self.video_encoder_name == "clip_1frame":
            mid_frame_index = int(video.shape[2] / 2)
            frame = video[:, :, mid_frame_index, :, :].contiguous()
            video_features = self.get_img_embeddings(frame)
        elif self.video_encoder_name == "clip_avg_frames":
            start_frame_index = 0
            mid_frame_index = int(video.shape[2] / 2)
            last_frame_index = -1
            video_features = self.get_img_embeddings(video[:, :, start_frame_index, :, :])
            video_features += self.get_img_embeddings(video[:, :, mid_frame_index, :, :])
            video_features += self.get_img_embeddings(video[:, :, last_frame_index, :, :])
        return video_features


class Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=2,
                bias=False,
            ),
            torch.nn.MaxPool1d(kernel_size=3),
        )

    def forward(self, batch):
        return self.net(batch)


class MW2StackRNNPooling(nn.Module):
    def __init__(self, input_dim=32, size_embeddings=256):
        super().__init__()
        self.name = MW2StackRNNPooling
        self.conv_layers = torch.nn.Sequential(
            torch.nn.GroupNorm(2, 6),
            Block(6, input_dim, 10),
            Block(input_dim, input_dim, 5),
            Block(input_dim, input_dim, 5),
            torch.nn.GroupNorm(4, input_dim),
        )
        self.gru = torch.nn.GRU(
            batch_first=True,
            input_size=input_dim,
            hidden_size=size_embeddings
        )

    def forward(self, batch):
        x = self.conv_layers(batch)
        x = x.permute(0, 2, 1)
        _, hidden = self.gru(x)
        return hidden[0]

