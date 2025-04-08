import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ---------------- 1. Patch Embedding (2D) + Time Flatten ---------------
class ViViTPatchEmbed(nn.Module):
    """
    将输入视频 [B, T, C, H, W] 拆分成时空 token。
    此处仅演示“对每帧做2D Patch Embedding，然后把时间维度展开”。
    """
    def __init__(self, in_channels=3, patch_size=16, emb_dim=768,
                 image_size=(128,128)):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.image_size = image_size
        
        # 计算行列方向上各有多少patch
        self.num_patches_h = image_size[0] // patch_size
        self.num_patches_w = image_size[1] // patch_size
        self.num_patches_per_frame = self.num_patches_h * self.num_patches_w
        
        # 线性投影, 输入patch大小 = patch_size * patch_size * in_channels
        patch_dim = patch_size * patch_size * in_channels
        self.proj = nn.Linear(patch_dim, emb_dim)
        
    def forward(self, x):
        """
        x: [B, T, C, H, W]
        return: patch_tokens, shape [B, T * num_patches_per_frame, emb_dim]
        """
        B, T, C, H, W = x.shape
        # 检查分辨率是否能整除
        assert H == self.image_size[0] and W == self.image_size[1], \
            "Input image_size must match the predefined patch embedding size"
        
        # reshape => [B*T, C, H, W]
        x = x.view(B*T, C, H, W)
        
        # 按 (patch_size, patch_size) 切分
        patches = self._reshape_as_patches(x)
        # => [B*T, num_patches, patch_size^2 * C]
        # num_patches = num_patches_per_frame
        
        # linear投影 => [B*T, num_patches, emb_dim]
        patch_tokens = self.proj(patches)
        
        # reshape => [B, T, num_patches, emb_dim]
        patch_tokens = patch_tokens.view(B, T, self.num_patches_per_frame, self.emb_dim)
        
        # 合并T和num_patches => [B, T * num_patches, emb_dim]
        patch_tokens = patch_tokens.view(B, T*self.num_patches_per_frame, self.emb_dim)
        return patch_tokens
    
    def _reshape_as_patches(self, x):
        """
        x: [B*T, C, H, W]
        => patches: [B*T, num_patches, patch_size^2 * C]
        """
        BxT, C, H, W = x.shape
        ph = pw = self.patch_size
        # 分块
        # (方案1) unfold操作
        unfold = nn.Unfold(kernel_size=(ph, pw), stride=(ph, pw))
        patches = unfold(x)  # => [B*T, C*ph*pw, num_patches_h * num_patches_w]
        patches = patches.permute(0, 2, 1)  # => [B*T, num_patches, patch_dim]
        return patches


# ---------------- 2. 时空位置编码 (可简单相加) ----------------
class ViViTPosEmbed(nn.Module):
    """
    对[batch, num_token, emb_dim] 的序列加可学习的位置编码
    (这里忽略时间与空间的分离，简单地对 tokens 序列加1D位置编码)
    """
    def __init__(self, max_len=1000, emb_dim=768):
        super().__init__()
        self.position_embedding = nn.Parameter(torch.zeros(1, max_len, emb_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
    
    def forward(self, x):
        # x: [B, N, emb_dim], N <= max_len
        B, N, D = x.shape
        # add => [1, N, D]
        pos_emb = self.position_embedding[:, :N, :]
        return x + pos_emb


# ---------------- 3. Transformer Encoder (多层) ----------------
class TransformerEncoderLayer(nn.Module):
    """
    标准 Transformer Encoder 单层
    """
    def __init__(self, emb_dim=768, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.ln2 = nn.LayerNorm(emb_dim)
        self.mlp_hidden_dim = int(emb_dim*mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, self.mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(self.mlp_hidden_dim, emb_dim),
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [B, N, D]
        # 1) Multi-head self attention
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)
        
        # 2) Feed forward
        x_norm = self.ln2(x)
        ff_out = self.mlp(x_norm)
        x = x + self.dropout(ff_out)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, emb_dim=768, num_heads=8, mlp_ratio=4.0, dropout=0.1, depth=6):
        super().__init__()
        layers = []
        for _ in range(depth):
            layer = TransformerEncoderLayer(emb_dim, num_heads, mlp_ratio, dropout)
            layers.append(layer)
        self.blocks = nn.ModuleList(layers)
        self.ln_final = nn.LayerNorm(emb_dim)
    
    def forward(self, x):
        # x: [B, N, D]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_final(x)
        return x

# ---------------- 4. Decoder/Head for video prediction ---------------
class SimpleVideoHead(nn.Module):
    """
    简化的“解码器”：
    - 将最后的 token 序列 => MLP => reshape 成 [B, T, C, H, W] (同输入维度)
    - 仅作演示，真实Video Prediction可能需更复杂的反卷积(ConvTranspose3D)等
    """
    def __init__(self, emb_dim=768, num_patches=16, 
                 out_frames=8, out_channels=3, patch_size=16,
                 image_size=(128,128)):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_patches = num_patches
        self.out_frames = out_frames
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.image_size = image_size
        
        # 计算 height_patches, width_patches
        self.num_patch_h = image_size[0] // patch_size
        self.num_patch_w = image_size[1] // patch_size
        # total patch = num_patch_h * num_patch_w
        
        # 线性映射: emb_dim -> patch像素
        # 让我们输出 = (out_channels * patch_size^2)
        self.out_patch_dim = out_channels * patch_size * patch_size
        
        self.proj = nn.Linear(emb_dim, self.out_patch_dim)
        
    def forward(self, x):
        """
        x: [B, N, emb_dim], N = out_frames * (num_patch_h * num_patch_w)
        return: [B, out_frames, out_channels, H, W]
        """
        B, N, D = x.shape
        
        # 线性 => [B, N, out_patch_dim]
        patch_vals = self.proj(x)  # => [B, N, out_patch_dim]
        
        # reshape => [B, out_frames, num_patches, out_channels, patch_size^2]
        # N should = out_frames * (num_patch_h * num_patch_w)
        # note: num_patches = num_patch_h * num_patch_w
        patches_per_frame = self.num_patch_h * self.num_patch_w
        # check N
        assert N == self.out_frames * patches_per_frame, \
            f"N mismatch, got {N}, expect {self.out_frames}*{patches_per_frame}"
        
        patch_vals = patch_vals.view(B, self.out_frames, patches_per_frame, 
                                     self.out_channels, self.patch_size*self.patch_size)
        
        # reshape 每个patch像素 => [patch_size, patch_size]
        patch_vals = patch_vals.view(B, self.out_frames, patches_per_frame, 
                                     self.out_channels, self.patch_size, self.patch_size)
        
        # 然后按patch组合回每帧 (num_patch_h, num_patch_w)
        # patches_per_frame = num_patch_h * num_patch_w
        patch_vals = patch_vals.view(B, self.out_frames, 
                                     self.num_patch_h, self.num_patch_w,
                                     self.out_channels, self.patch_size, self.patch_size)
        # reorder => [B, out_frames, out_channels, (num_patch_h*patch_size), (num_patch_w*patch_size)]
        out_frames_data = patch_vals.permute(0,1,4,2,5,3,6).contiguous()
        # => [B, T, C, (num_patch_h*ph), (num_patch_w*pw)]
        out_frames_data = out_frames_data.view(B, 
                                               self.out_frames, 
                                               self.out_channels,
                                               self.num_patch_h*self.patch_size,
                                               self.num_patch_w*self.patch_size)
        
        return out_frames_data


# ---------------- 5. Assemble: ViViT model for Video Prediction ---------------
class ViViTVideoPredictor(nn.Module):
    """
    1) Patch Embedding -> position embed
    2) Transformer Encoder
    3) A simple head => reconstruct frames
    """
    def __init__(self, 
                 in_channels=3,
                 image_size=(128,128),
                 patch_size=16,
                 emb_dim=256,
                 depth=4,
                 num_heads=4,
                 mlp_ratio=4.0,
                 out_channels=3,
                 in_frames=8,
                 out_frames=8):
        super().__init__()
        
        self.patch_embed = ViViTPatchEmbed(in_channels=in_channels,
                                           patch_size=patch_size,
                                           emb_dim=emb_dim,
                                           image_size=image_size)
        
        max_len = in_frames*self.patch_embed.num_patches_per_frame*2  # some big enough
        self.pos_embed = ViViTPosEmbed(max_len=max_len, emb_dim=emb_dim)
        
        self.encoder = TransformerEncoder(emb_dim=emb_dim,
                                          num_heads=num_heads,
                                          mlp_ratio=mlp_ratio,
                                          dropout=0.1,
                                          depth=depth)
        
        # 这里假定: 输入in_frames => 输出 out_frames
        self.head = SimpleVideoHead(emb_dim=emb_dim,
                                    num_patches=self.patch_embed.num_patches_per_frame,
                                    out_frames=out_frames,
                                    out_channels=out_channels,
                                    patch_size=patch_size,
                                    image_size=image_size)
        
        self.in_frames = in_frames
        self.out_frames= out_frames
        self.patches_per_frame = self.patch_embed.num_patches_per_frame
    
    def forward(self, x):
        """
        x: [B, in_frames, in_channels, H, W]
        => returns: [B, out_frames, out_channels, H, W]
        """
        B, T, C, H, W = x.shape
        assert T == self.in_frames, f"input frames mismatch, got {T}, want {self.in_frames}"
        
        # 1) patch embed => [B, T*num_patches, emb_dim]
        patch_tokens = self.patch_embed(x)
        
        # 2) add positional embedding => [B, T*num_patches, emb_dim]
        patch_tokens = self.pos_embed(patch_tokens)
        
        # 3) transformer encoder => [B, T*num_patches, emb_dim]
        enc_out = self.encoder(patch_tokens)
        
        # 4) decode => [B, out_frames, out_channels, H, W]
        # 这里需要注意: N = out_frames * patch_per_frame
        # 所以做一个 trick: enc_out形状  => (B, N, D)
        #  *  N 需 = out_frames * patch_per_frame
        # 但我们目前 N = in_frames * patch_per_frame
        # => simplest approach:  assume out_frames = in_frames, just do reconstruction
        #    or define something like duplication or conv1d if we want different frames
        #    这里就简单 case: out_frames = in_frames
        #    => N= in_frames * patch_per_frame
        #         out_frames = in_frames
        
        if self.out_frames != self.in_frames:
            # 这里如果 out_frames != in_frames, 需更复杂处理, 
            # 例如让 transformer输出time-latent => decode to out_frames
            # 省略
            raise NotImplementedError("This simple example only handles out_frames == in_frames.")
        
        # => shape [B, in_frames * patch_per_frame, emb_dim]
        out_vid = self.head(enc_out)
        return out_vid


# ------------------- 测试脚本 -------------------
if __name__=="__main__":
    # 生成假数据
    B = 2
    in_frames = 4
    C = 3
    H, W = 128, 128
    x = torch.randn(B, in_frames, C, H, W)
    
    model = ViViTVideoPredictor(in_channels=3,
                                image_size=(128,128),
                                patch_size=16,
                                emb_dim=256,
                                depth=4,
                                num_heads=4,
                                mlp_ratio=4.0,
                                out_channels=3,
                                in_frames=4,
                                out_frames=4)
    
    y = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    # 期望: [B, out_frames, out_channels, H, W] => [2, 4, 3, 128, 128]