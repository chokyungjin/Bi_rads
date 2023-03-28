import torch as t
import torch.nn as nn


class ACMBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 ops='minus'):
        super(ACMBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.ops = ops
        self.k_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, (1,1), groups=23),
        )

        self.q_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, (1,1), groups=23),
        )

        self.global_pooling = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels//2, (1,1)),
            nn.ReLU(),
            nn.Conv2d(self.out_channels//2, self.out_channels, (1,1)),
            nn.Sigmoid()
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.normalize = nn.Softmax(dim=3)

    def _get_normalized_features(self, x):
        ''' Get mean vector by channel axis '''
        
        c_mean = self.avgpool(x)
        return c_mean

    def _get_orth_loss(self, K, Q):
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        orth_loss = cos(K, Q)
        orth_loss = t.mean(orth_loss, dim=0)
        return orth_loss
    
    def _get_orth_loss_ACM(self, K, Q, c):
        orth_loss = t.mean(K*Q/c, dim=1, keepdim=True)
        return orth_loss
    
    def forward(self, x1, x2):
        mean_x1 = self._get_normalized_features(x1)
        mean_x2 = self._get_normalized_features(x2)
        x1_mu = x1-mean_x1
        x2_mu = x2-mean_x2
        
        K = self.k_conv(x1_mu)
        Q = self.q_conv(x2_mu)

        b, c, h, w = K.shape

        K = K.view(b, c, 1, h*w)
        K = self.normalize(K)
        K = K.view(b, c, h, w)

        Q = Q.view(b, c, 1, h*w)
        Q = self.normalize(Q)
        Q = Q.view(b, c, h, w)

        K = t.einsum('nchw,nchw->nc',[K, x1_mu])
        Q = t.einsum('nchw,nchw->nc',[Q, x2_mu])
        K = K.view(K.shape[0], K.shape[1], 1, 1)
        Q = Q.view(Q.shape[0], Q.shape[1], 1, 1)

        channel_weights1 = self.global_pooling(mean_x1)
        channel_weights2 = self.global_pooling(mean_x2)
        
        if self.ops == 'minus':
            out1 = x1 + K - Q
            out2 = x2 + K - Q
        else:
            out1 = x1 + K + Q
            out2 = x2 + K + Q
        
        out1 = channel_weights1 * out1
        out2 = channel_weights2 * out2
        
        orth_loss = self._get_orth_loss(K,Q)

        return out1, out2, orth_loss