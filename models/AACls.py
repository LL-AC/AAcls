import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
from einops import rearrange
from timm.models.layers import *
from torch.jit import Final
from transformers import AutoTokenizer, AutoModel



class MoHAttention(nn.Module):
    # https://arxiv.org/abs/2410.11842
    # MoH: Multi-Head Attention as Mixture-of-Head Attention
    fused_attn: Final[bool]
    LOAD_BALANCING_LOSSES = []

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
            shared_head=0,
            routed_head=0,
            head_dim=None,
    ):
        super().__init__()
        # assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        
        if head_dim is None:
            self.head_dim = dim // num_heads
        else:
            self.head_dim = head_dim
        
        self.scale = self.head_dim ** -0.5
        #self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, (self.head_dim * self.num_heads) * 3, bias=qkv_bias)
        
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.head_dim * self.num_heads, dim)
        
        self.proj_drop = nn.Dropout(proj_drop)

        self.shared_head = shared_head
        self.routed_head = routed_head
        
        if self.routed_head > 0:
            self.wg = torch.nn.Linear(dim, num_heads - shared_head, bias=False)
            if self.shared_head > 0:
                self.wg_0 = torch.nn.Linear(dim, 2, bias=False)

        if self.shared_head > 1:
            self.wg_1 = torch.nn.Linear(dim, shared_head, bias=False)

    def forward(self, x):
        B, N, C = x.shape

        _x = x.reshape(B * N, C)
        
        if self.routed_head > 0:
            logits = self.wg(_x)
            gates = F.softmax(logits, dim=1)

            num_tokens, num_experts = gates.shape
            _, indices = torch.topk(gates, k=self.routed_head, dim=1)
            mask = F.one_hot(indices, num_classes=num_experts).sum(dim=1)

            if self.training:
                me = gates.mean(dim=0)
                ce = mask.float().mean(dim=0)
                l_aux = torch.mean(me * ce) * num_experts * num_experts

                MoHAttention.LOAD_BALANCING_LOSSES.append(l_aux)

            routed_head_gates = gates * mask
            denom_s = torch.sum(routed_head_gates, dim=1, keepdim=True)
            denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
            routed_head_gates /= denom_s
            routed_head_gates = routed_head_gates.reshape(B, N, -1) * self.routed_head

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # if self.fused_attn:
        #     x = F.scaled_dot_product_attention(
        #         q, k, v,
        #         dropout_p=self.attn_drop.p if self.training else 0.,
        #     )
        # else:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        
        if self.routed_head > 0:
            x = x.transpose(1, 2)

            if self.shared_head > 0:
                shared_head_weight = self.wg_1(_x)
                shared_head_gates = F.softmax(shared_head_weight, dim=1).reshape(B, N, -1) * self.shared_head

                weight_0 = self.wg_0(_x)
                weight_0 = F.softmax(weight_0, dim=1).reshape(B, N, 2) * 2
        
                shared_head_gates = torch.einsum("bn,bne->bne", weight_0[:,:,0], shared_head_gates)
                routed_head_gates = torch.einsum("bn,bne->bne", weight_0[:,:,1], routed_head_gates)
                
                masked_gates = torch.cat([shared_head_gates, routed_head_gates], dim=2)
            else:
                masked_gates = routed_head_gates

            x = torch.einsum("bne,bned->bned", masked_gates, x)
            x = x.reshape(B, N, self.head_dim * self.num_heads)
        else:
            shared_head_weight = self.wg_1(_x)
            masked_gates = F.softmax(shared_head_weight, dim=1).reshape(B, N, -1) * self.shared_head
            x = x.transpose(1, 2)

            x = torch.einsum("bne,bned->bned", masked_gates, x)
            x = x.reshape(B, N, self.head_dim * self.num_heads)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        #self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn
    
class Block(nn.Module):
    def __init__(self, dim, num_heads=12, mlp_ratio=4.,drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, shared_head=2, routed_head=6):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # self.attn = Attention(
        #     dim,
        #     num_heads=num_heads,
        #     qkv_bias=True,
        #     qk_norm=True,
        #     attn_drop=0.1,
        #     proj_drop=0.1,
        #     norm_layer=norm_layer,
        # )
        self.attn = MoHAttention(dim, shared_head=2, routed_head=6)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn




class AACls(nn.Module):
    def __init__(self, config, n_classes):
        super(AACls, self).__init__()

        self.config = config

        self.feature_extractor = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.feature_extractor.fc = nn.Sequential(nn.Linear(512, 768), nn.ReLU())

        #freeze in training
        tokenizer = AutoTokenizer.from_pretrained("pubmedbert")
        self.text_token = tokenizer(config['text_prompt'], padding=True, truncation=True, return_tensors='pt')
        self.text_encoder = AutoModel.from_pretrained("pubmedbert")
        with torch.no_grad():
            output = self.text_encoder(**self.text_token)
            x = output[0]
        self.text_features = x[torch.arange(x.shape[0]), self.text_token['input_ids'].argmax(dim=-1)] 

        self.dim = 768

        self.a = nn.Parameter(torch.tensor(1.0))

        self.projector = nn.Sequential(nn.Linear(768, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, 768)) #!
        

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        # 多层 Transformer
        self.layer = Block(dim=self.dim, num_heads=8)

       #self.MIL_f = MIL_f
        self.n_classes = n_classes

        self.norm = nn.LayerNorm(self.dim)
        self.fc = nn.Linear(self.dim, n_classes)

        self.ce = nn.CrossEntropyLoss()
        self.semi_ce = nn.CrossEntropyLoss()

    
    def MFS(self, ins_feat, prompts):

        #no choice
        if len(ins_feat) == 0:
            return ins_feat, None, None

        feats = F.normalize(ins_feat, p=2, dim=-1)  #shape [n d]
        pro = F.normalize(prompts, p=2, dim=-1)  #shape [c d]

        text_relevance = feats @ pro.t()    #shape [n c]

        #print(targets)
        relevance = F.softmax(text_relevance/0.01, dim=-1)

        relevance, _ = relevance.max(dim=-1)
        
        n, _ = text_relevance.shape
        if n == 1:
            threshold = 0
        else:
            threshold = self.a*relevance.mean()

        selected_indices = relevance >= threshold
        
        selected_features = ins_feat[selected_indices]   # [n d]
        
        return selected_features, selected_indices, text_relevance
    
    
    def PPL(self, s_fi, t_r, target_class):
        
        if t_r is None:
            return 0
        # normal
        device = t_r.device       
    
        pred = t_r[s_fi]
        if len(pred) == 0:
            return 0

        n, _ = pred.shape

        t_c = torch.full((n,), target_class).to(device)
        loss = self.semi_ce(pred/0.07, t_c)

        return loss
        
    def record(self, anomalies_matrix, mask):

        anomalies_matrix = anomalies_matrix.clone().detach()  # 创建一个副本，避免修改原始矩阵

        anomalies_matrix = F.softmax(anomalies_matrix/0.01, dim=-1)       

        # 设置不需要保留的行和列
        if mask is not None:
            mask = mask.clone().detach()

            #取图的下标、 对应图的异常分数
            return {'select': mask.cpu().numpy().tolist(), 
                    'plane_idx': np.arange(len(mask)).tolist(),
                    'anomalies_matrix': anomalies_matrix.cpu().numpy().tolist()}
        return {'select': '',
                'anomalies_matrix': anomalies_matrix.cpu().numpy().tolist()} 
    
    
    def forward(self, x, len_list, labels):

        x = self.feature_extractor(x)

        ini_idx = 0

        y = []
        anomlies_socre = []
        patch_loss = torch.tensor(0.).to(x.device)

        for i, length in enumerate(len_list):
            ins_feat = x[ini_idx : ini_idx + length]
            ini_idx += length

            s_f, s_fi, t_r = self.MFS(ins_feat, self.text_features)

            
            record = [self.record(t_r, s_fi)]
            
            anomlies_socre.append(record)


            #PPL
            patch_loss += self.PPL(s_fi, t_r, labels[i])

            #filter converge
            h = s_f
            h = h.unsqueeze(0) 

            B, N, _ = h.shape
            cls_tokens = self.cls_token.expand(B, -1, -1)
            h = torch.cat((cls_tokens, h), dim=1)

            # Transformer 整体聚合
            h, _ = self.layer(h)
            h = self.norm(h)

            y.append(h[:, 0])

            
        h = torch.cat(y, dim=0)    #h [B dim]

        logits = self.fc(h) #[B, n_classes]

        print(logits.shape, )

        cls_loss = self.ce(logits, labels)
        #print(cls_loss, patch_loss)

        loss = (1.0 * cls_loss) + 0.1*(patch_loss / len(len_list))

        results_dict = {'predicted': logits, 
                        'loss': loss,
                        'anomalies_score': anomlies_socre,  # 用于调试和可解释性分析,
                        'features': h,   #t-SNE  可视化
                        }        
        return results_dict


config = {
    'text_prompt':
    [   'Omphalocele is a congenital anomaly caused by the failure of midline skin folding during embryonic development, resulting in a midline abdominal wall defect where abdominal organs and peritoneum protrude externally. On ultrasound, it appears as a midline mass extending outward from the umbilical root, covered by a membranous sac with well-defined margins, and the umbilical cord is attached to the apex of the mass.',
        'Gastroschisis is a congenital defect caused by incomplete development of the anterior abdominal wall during the embryonic period. It is typically characterized by a full-thickness abdominal wall defect, most commonly to the right of the umbilicus, while the umbilical cord remains normally attached. On ultrasound, it appears as an interruption of the echogenic abdominal wall line with herniation of abdominal organs directly into the amniotic fluid without a covering membrane. The protruding structures are primarily bowel loops, which may show peristalsis within the amniotic fluid.',
        'Duodenal Atresia or Stenosis is one of the most common causes of small bowel obstruction, primarily attributed to a failure in the recanalization process during embryonic development. On ultrasound, it is characterized by marked dilatation of the stomach and the proximal duodenum, forming the classic "double bubble sign" in the fetal transverse abdomen. The left-sided bubble represents the stomach, while the middle or right-sided bubble corresponds to the distended proximal duodenum, with a visible connection between them at the pyloric canal.',
        'Renal Agenesis is caused by the failure of ureteric bud development, preventing the induction of metanephric differentiation into the kidneys. On ultrasound, it is characterized by the absence of renal structures in the renal fossae and surrounding areas on both sides of the spine. Bilateral renal agenesis is often accompanied by the "lying-down adrenal sign," where the adrenal glands appear flattened. Additional diagnostic indicators include oligohydramnios or anhydramnios from the mid-pregnancy stage and associated pulmonary hypoplasia.',
        'Multicystic Dysplastic Kidney (MCDK) is a congenital renal anomaly characterized by the presence of multiple cysts of varying sizes that do not communicate with the renal pelvis, most commonly affecting one kidney. On ultrasound, the affected kidney appears as a cluster of multiple anechoic cystic structures of different sizes within the renal parenchyma. Normal renal cortex is either absent or only partially preserved, and the collecting system is not identifiable.',
        'Normal Fetal Ultrasound is characterized by the presence of all expected anatomical structures without abnormalities. On ultrasound, the abdominal wall is intact, with no evidence of defects such as omphalocele or gastroschisis. The stomach and proximal duodenum appear normal, with no signs of dilation or the "double bubble sign" indicative of duodenal atresia. Both kidneys are clearly visualized in their respective renal fossae with normal echogenicity and morphology, ruling out renal agenesis. The amniotic fluid volume is within normal limits, and no associated anomalies are detected.'
    ]
}


if __name__ == '__main__':
    model = AACls(config, 6)

    x = torch.randn((100,3,224,224))
    labels = torch.tensor([1,0,1,1,2,3,0])
    
    #len_list is case images
    y = model(x, [20,40,10,3,7,10,10], labels)
    print(y['predicted'].shape)
