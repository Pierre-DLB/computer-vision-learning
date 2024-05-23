import torch
from torch import nn
import torchvision
import torchvision.transforms as T

# /!\ SET DEVICE

class SelfAttention(nn.Module):
    def __init__(self, D_input, D_h):
        super().__init__()
        self.D_h = D_h
        self.D_input=D_input

        self.q_mat = nn.Linear(in_features=self.D_input, out_features=self.D_h, bias=None)
        self.k_mat = nn.Linear(in_features=self.D_input, out_features=self.D_h, bias=None)
        self.v_mat = nn.Linear(in_features=self.D_input, out_features=self.D_h, bias=None)
    
    def forward(self, z):
        q, k, v = self.q_mat(z), self.k_mat(z), self.v_mat(z)
        A = torch.softmax(torch.matmul(q, torch.transpose(k, 1, 0)) / torch.sqrt(torch.tensor(self.D_h)), axis=1)
        return torch.matmul(A, v)
        

class MSA(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.fc_out = nn.Linear(embedding_dim, embedding_dim)
        # for q, k, v, instead of having k linear layers acting in parallel with shape nn.Linear(embedding_dim, embedding_dim/k), 
        # they are all stacked together in one layer, and we cut the output afterwards.



    def forward(self, x):
        N, seq_length, embedding_dim = x.shape

        # Split the embedding into self.num_heads different pieces
        # view : see the vector reshaped as "", without copying i.e. do specific operations easily
        queries = self.query(x).view(N, seq_length, self.num_heads, self.head_dim)
        keys = self.key(x).view(N, seq_length, self.num_heads, self.head_dim)
        values = self.value(x).view(N, seq_length, self.num_heads, self.head_dim)

        # Transpose to get dimensions (N, num_heads, seq_length, head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Calculate the attention scores
        scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        attention = torch.softmax(scores, dim=-1)

        # Get the weighted values
        out = torch.matmul(attention, values)

        # Reshape to (N, seq_length, embedding_dim)
        out = out.transpose(1, 2).contiguous().view(N, seq_length, embedding_dim)

        # Apply the final linear layer (unification layer)
        out = self.fc_out(out)
        return out
    

    # class MSA(nn.Module): # PROBLEM WITH MEMORY MANAGEMENT BECAUSE OF THE LIST#
    #   def __init__(self, embedding_dim, num_heads):
    #         super().__init__()
    #         self.k = num_heads
    #         self.D_input = embedding_dim
    #         self.D_h = embedding_dim//num_heads

    #         self.attentions = [SelfAttention(self.D_h, self.D_h) for i in range(self.k)]
    #         self.unification_matrix = nn.Linear(self.D_input, self.D_input, bias=None)
        
    #     def forward(self, z):
    #         vectors = torch.split(z, split_size_or_sections=self.D_h, dim=1)

    #         for i in range(self.k):
    #             vectors[i] = self.attentions[i](vectors[i])

    #         MSA = torch.cat(vectors, dim=1)
    #         return self.unification_matrix(MSA)


class MLP(nn.Module):
    def __init__(self, embedding_dim, mlp_size, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mlp_size = mlp_size
        
        self.block = nn.Sequential(nn.LayerNorm(normalized_shape=self.embedding_dim),
                        nn.Linear(in_features=embedding_dim, out_features=mlp_size),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(in_features=mlp_size, out_features=embedding_dim),
                        nn.Dropout(dropout))

    def forward(self, z):
        return self.block(z)


class Encoder(nn.Module):
    def __init__(self,
                embedding_dim:int=768,
                num_heads:int=12,
                mlp_size:int=3072, 
                mlp_dropout:float=0.1, 
                attn_dropout:float=0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_size = mlp_size
        self.mlp_dropout = mlp_dropout
        self.attn_dropout = attn_dropout
        self.norm = nn.LayerNorm(normalized_shape=self.embedding_dim)
        self.MSA = MSA(self.embedding_dim, self.num_heads)
        self.MLP = MLP(embedding_dim=self.embedding_dim, mlp_size=self.mlp_size, dropout=self.mlp_dropout)

    
    def forward(self, x):
        att_x = self.MSA(self.norm(x)) + x
        mlp_x = self.MLP(self.norm(att_x)) + x 
        return mlp_x


class PatchEmbedder(nn.Module):

    def __init__(self, patch_size: int, image_size: int=224, embedding_dim:int = 768, random=True):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=3, out_channels=embedding_dim, 
                              kernel_size=(patch_size, patch_size), stride=patch_size, padding=0)
                            # same as cutting in smaller patches
        self.flat = nn.Flatten(start_dim=2, end_dim=3)
        patch_num=(image_size//patch_size)**2

        # ADD [CLASS] TOKEN
        if random:
            self.class_embedding= nn.Parameter(torch.rand((1, 1, embedding_dim)), requires_grad=True)
        else:
            self.class_embedding= nn.Parameter(torch.ones((1, 1, embedding_dim)), requires_grad=True)

        # ADD POSITION EMBEDDING
        if random:
            self.pos_embedding= nn.Parameter(torch.ones((1, patch_num+1 , embedding_dim)), requires_grad=True)
        else:
            self.pos_embedding= nn.Parameter(torch.rand((1, patch_num+1 , embedding_dim)), requires_grad=True)
        # Add the position embedding to the patch and class token embedding

    def forward(self, x):
        batch_size = x.shape[0]
        y = self.flat(self.conv(x))
        y = y.permute(0, 2, 1)
        class_token = self.class_embedding.expand(batch_size, -1, -1)
        class_pos_emb = torch.cat((class_token, y),dim=1) + self.pos_embedding

        return class_pos_emb


class ViT(nn.Module):
    def __init__(self,
                    img_size:int=224, # Training resolution from Table 3 in ViT paper
                    in_channels:int=3, # Number of channels in input image
                    patch_size:int=16, # Patch size
                    num_transformer_layers:int=12, # Layers from Table 1 for ViT-Base
                    embedding_dim:int=768, # Hidden size D from Table 1 for ViT-Base
                    mlp_size:int=3072, # MLP size from Table 1 for ViT-Base
                    num_heads:int=12, # Heads from Table 1 for ViT-Base
                    attn_dropout:float=0, # Dropout for attention projection
                    mlp_dropout:float=0.1, # Dropout for dense/MLP layers
                    embedding_dropout:float=0.1, # Dropout for patch and position embeddings
                    num_classes:int=1000): # Default for ImageNet but can customize this
        super().__init__() # don't forget the super().__init__()!
        assert img_size % patch_size == 0, f"Image size must be divisible by patch size, image size: {img_size}, patch size: {patch_size}."


        self.Embedding = PatchEmbedder(patch_size=patch_size,
                                       image_size=img_size,
                                       embedding_dim=embedding_dim, random=True)
        
        self.EmbeddingDropout = nn.Dropout(p=embedding_dropout)
        
        self.StackedEncoders = nn.Sequential(*[Encoder(embedding_dim=embedding_dim,
                                                num_heads=num_heads,
                                                mlp_size=mlp_size,
                                                mlp_dropout=mlp_dropout,
                                                attn_dropout=attn_dropout)
                                        for layer in range(num_transformer_layers)])
        self.Classifier = nn.Sequential(
        nn.LayerNorm(normalized_shape=embedding_dim),
        nn.Linear(in_features=embedding_dim, out_features=num_classes)
        )

    def forward(self, x):
        embedded_x = self.Embedding(x)
        embedded_x = self.EmbeddingDropout(embedded_x)
        embedded_x = self.StackedEncoders(embedded_x)
        class_x = self.Classifier(embedded_x)
        return class_x[:,0]

            



