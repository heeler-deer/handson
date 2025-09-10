from torch import nn
import torch
import numpy as np
class ScaledDotProductAttention(nn.Module):
    def __init__(self,scale):
        super().__init__()
        self.scale=scale
        self.softmax=nn.Softmax(dim=2)
    def forward(self,q,k,v,mask=None):
        u=torch.bmm(q,k.transpose(1,2))
        u=u/self.scale
        if mask is not None:
            u=u.masked_fill(mask,-np.inf)
        attn=self.softmax(u)
        output=torch.bmm(attn,v)
        return attn,output
    
    
    

class MultiHeadAttention(nn.Module):
    def __init__(self,d_k_,d_v_,d_k,d_v,d_o,n_head):
        super().__init__()
        self.n_head=n_head
        self.d_k=d_k
        self.d_v=d_v
        
        self.fc_q=nn.Linear(d_k_,n_head*d_k)
        self.fc_k=nn.Linear(d_k_,n_head*d_k)
        self.fc_v=nn.Linear(d_v_,n_head*d_v)
        
        self.attention=ScaledDotProductAttention(scale=np.power(d_k,0.5))
        self.fc_o=nn.Linear(n_head*d_v,d_o)
        
        
    def forward(self,q,k,v,mask=None):
        n_head,d_q,d_k,d_v=self.n_head,self.d_k,self.d_k,self.d_v
        batch,n_q,d_q_=q.size()
        batch,n_k,d_k_=k.size()
        batch,n_v,d_v_=v.size()
        
        
        
        q=self.fc_q(q)
        k=self.fc_k(k)
        v=self.fc_v(v)
        
        q=q.view(batch,n_q,n_head,d_q).permute(2,0,1,3).contiguous().view(-1,n_q,d_q)
        k=k.view(batch,n_k,n_head,d_k).permute(2,0,1,3).contiguous().view(-1,n_k,d_k)
        v=v.view(batch,n_v,n_head,d_v).permute(2,0,1,3).contiguous().view(-1,n_v,d_v)
        
        if mask is not None:
            mask=mask.repeat(n_head,1,1)
        attn,output=self.attention(q,k,v,mask=mask)
        output=output.view(n_head,batch,n_q,d_v).permute(1,2,0,3).contiguous().view(batch,n_q,-1)
        output=self.fc_o(output)
        return attn,output







class SelfAttention(nn.Module):
    """ Self-Attention """

    def __init__(self, n_head, d_k, d_v, d_x, d_o):
        super().__init__()
        self.wq = nn.Parameter(torch.Tensor(d_x, d_k))
        self.wk = nn.Parameter(torch.Tensor(d_x, d_k))
        self.wv = nn.Parameter(torch.Tensor(d_x, d_v))

        self.mha = MultiHeadAttention(n_head=n_head, d_k_=d_k, d_v_=d_v, d_k=d_k, d_v=d_v, d_o=d_o)

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / np.power(param.size(-1), 0.5)
            param.data.uniform_(-stdv, stdv)

    def forward(self, x, mask=None):
        q = torch.matmul(x, self.wq)   
        k = torch.matmul(x, self.wk)
        v = torch.matmul(x, self.wv)

        attn, output = self.mha(q, k, v, mask=mask)

        return attn, output


if __name__ == "__main__":
    n_x = 4
    d_x = 80
    batch=1
    x = torch.randn(batch, n_x, d_x)
    mask = torch.zeros(batch, n_x, n_x).bool()

    selfattn = SelfAttention(n_head=8, d_k=128, d_v=64, d_x=80, d_o=80)
    attn, output = selfattn(x, mask=mask)

    print(attn.size())
    print(output.size())