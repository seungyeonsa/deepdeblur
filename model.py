import torch
import torchvision
import torch.nn as nn


class ResEncoder(nn.Module):
    def __init__(self, dim_in, dim_inter, dim_res):
        super(ResEncoder, self).__init__()
        
        self.pre_conv = nn.Conv2d(dim_in, dim_inter, kernel_size=5, padding=2)
        
        """ self.resblocks = nn.Sequential()
        for _ in range(19):
            self.resblocks.append(Residual_Block(dim_inter, dim_res)) """
        
        self.resblocks = []
        for _ in range(19):
            self.resblocks.append(Residual_Block(dim_inter, dim_res))
        self.resblocks = nn.Sequential(*self.resblocks)
                
        self.post_conv = nn.Conv2d(dim_inter,
                                   dim_in,
                                   kernel_size=5,
                                   padding=2)
        
        self.upsample = nn.ConvTranspose2d(
            dim_in, dim_in, kernel_size=5,
            padding=2, output_padding=1, stride=2
        )
        
        
    def forward(self, x):
        out = self.pre_conv(x)
        out = self.resblocks(out)
        out = self.post_conv(out)
        out_upsampled = self.upsample(out)
        
        return out, out_upsampled
        
class Residual_Block(nn.Module):
    def __init__(self, in_dim, mid_dim):
        super(Residual_Block, self).__init__()
        
        self.residual_block = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(mid_dim, in_dim, kernel_size=5, padding=2)
        )
        
        self.relu = nn.LeakyReLU()
    
    def forward(self,x):
        out = self.residual_block(x)
        out = out + x
        out = self.relu(out)
        return out
    

class MainModel(nn.Module): #Total model parameters: 23,387,829
    def __init__(self, dim_in, dim_inter, dim_res, **kwargs):
        super(MainModel, self).__init__()
        
        self.encoder1 = ResEncoder(dim_in, dim_inter, dim_res)
        self.encoder2 = ResEncoder(dim_in, dim_inter, dim_res)
        self.encoder3 = ResEncoder(dim_in, dim_inter, dim_res)
        
        """ for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight) """
                
    def forward(self, x1, x2, x3):
        out1, out1_upsampled = self.encoder1(x1)
        out2, out2_upsampled = self.encoder2(x2+out1_upsampled)
        out3, _ = self.encoder3(x3+out2_upsampled)
        return out1, out2, out3
        




if __name__ == '__main__':
    #import pdb; pdb.set_trace()
    # model = ResEncoder(3, 64, 256)
    model = MainModel(3, 64, 256)
    model_path = './models'
    save_path = '{}/trained_.model{}'.format(model_path, 1)
    torch.save(model.state_dict(), save_path)

        
    #print(model)
"""     model_in1 = torch.randn(4, 3, 64, 64)
    model_in2 = torch.randn(4, 3, 128, 128)
    model_in3 = torch.randn(4, 3, 256, 256)
    model_out1, model_out2, model_out3 = model(model_in1, model_in2, model_in3)
    print(model_in1.shape, model_in2.shape, model_in3.shape)
    print(model_out1.shape, model_out2.shape, model_out3.shape) """
    
