import torch
import torch.nn as nn
import torch.nn.functional as F


# for RFI flagging v1
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv3 = nn.Conv2d(1,2,kernel_size=(3,3),stride=1,padding=1,bias=False)
        self.conv5 = nn.Conv2d(1,2,kernel_size=(5,5),stride=1,padding=2,bias=False)
        
    def forward(self, x):
        x = self.conv3(x)>0
        x = x[0,0,:,:]+x[0,1,:,:]
        return x

net=Net()

aggressive_factor = [1.8,1.8] # (0,4] recommended, the larger the more aggressive
#aggressive_factor = [1.7,1.7] # (0,4] recommended, the larger the more aggressive



def init_net(net,aggressive_factor = [1.2,1.2]):
    
    # for lines
    kernel_vertical = torch.tensor([
        [-1,aggressive_factor[0],-1],
        [-1,aggressive_factor[0],-1],
        [-1,aggressive_factor[0],-1]]).T # transient

    kernel_horizontal = torch.tensor([
        [-1,-1,-1],
        [aggressive_factor[1],aggressive_factor[1],aggressive_factor[1]],
        [-1,-1,-1]]).T  # narrow band

    state_dict = net.state_dict()
    state_dict['conv3.weight'][0,0,:,:] = kernel_vertical[None,None,:,:]     # increase dimention of kernel
    state_dict['conv3.weight'][1,0,:,:] = kernel_horizontal[None,None,:,:]   # increase dimention of kernel
    net.load_state_dict(state_dict)

    return net

net  = init_net(net,aggressive_factor)


# for RFI flagging v1

class RFIconv(nn.Module):

    def __init__(self,device='cpu'):
        
        super(RFIconv, self).__init__()

        self.conv3 = nn.Conv2d(1,2,kernel_size=(3,3),stride=1,padding=1,bias=False)
        self.conv5 = nn.Conv2d(1,4,kernel_size=(5,5),stride=1,padding=2,bias=False)
        
        self.conv3_bool = nn.Conv2d(2,2,groups=2,kernel_size=(3,3),stride=1,padding=1,bias=False)
        self.conv5_bool = nn.Conv2d(4,4,groups=4,kernel_size=(5,5),stride=1,padding=2,bias=False)
        self.device = device

    def forward(self, x):

        dum0 = torch.tensor([0.0]).to(self.device)

        '''
        c3mask = torch.heaviside(F.conv2d(torch.heaviside(self.conv3(x),dum0),
            torch.ones([2,1,5,5]).to(self.device),bias=None,stride=1,padding=2,groups=2),dum0)

        c5mask = self.conv5_bool(torch.cat(
               (torch.heaviside(c5[0,0,:,:]-c3mask[0,0,:,:],dum0).squeeze()[None,None,:,:],
                torch.heaviside(c5[0,1,:,:]-c3mask[0,0,:,:],dum0).squeeze()[None,None,:,:],
                torch.heaviside(c5[0,2,:,:]-c3mask[0,1,:,:],dum0).squeeze()[None,None,:,:],
                torch.heaviside(c5[0,3,:,:]-c3mask[0,1,:,:],dum0).squeeze()[None,None,:,:]),1))
        '''
        y = (  torch.sum(self.conv3_bool(torch.heaviside(self.conv3(x),dum0)),dim=1,keepdim=True)
            +  torch.sum(self.conv5_bool(torch.heaviside(torch.heaviside(self.conv5(x),dum0)-torch.heaviside(F.conv2d(torch.heaviside(self.conv3(x),dum0),
            torch.ones([2,1,5,5]).to(self.device),bias=None,stride=1,padding=2,groups=2),dum0).repeat_interleave(2,dim=1),dum0))
            ,dim=1,keepdim=True) )>1e-7
        return y

"""
# with relu
    def forward(self, x):

        c3 = F.relu(self.conv3(x))
        c5 = F.relu(self.conv5(x))

        c3mask = torch.relu(F.conv2d(c3,
            torch.ones([2,1,5,5]),bias=None,stride=1,padding=2,groups=2))

        c5mask = self.conv5_bool(torch.cat(
               (F.relu(c5[0,0,:,:]-c3mask[0,0,:,:]*5).squeeze()[None,None,:,:],
                F.relu(c5[0,1,:,:]-c3mask[0,0,:,:]*5).squeeze()[None,None,:,:],
                F.relu(c5[0,2,:,:]-c3mask[0,1,:,:]*5).squeeze()[None,None,:,:],
                F.relu(c5[0,3,:,:]-c3mask[0,1,:,:]*5).squeeze()[None,None,:,:]),1))>1e-8

        y = (  torch.sum(self.conv3_bool(c3),dim=1,keepdim=True)
            +  torch.sum(c5mask,dim=1,keepdim=True) )>1e-8
        return y
"""


aggressive_factor =[1.6,1.65,0.5,0.5] # (0,4] recommended, the larger the more aggressive
#aggressive_factor = [1.7,1.7] # (0,4] recommended, the larger the more aggressive


def init_RFIconv(net,aggressive_factor = [1.6,1.65,0.5,0.5],device='cpu'):
    af = aggressive_factor
    # for lines
    kernel_vertical = torch.tensor([
        [-1,af[0],-1],
        [-1,af[0],-1],
        [-1,af[0],-1]]).T # transient

    kernel_horizontal = torch.tensor([
        [-1,-1,-1],
        [af[1],af[1],af[1]],
        [-1,-1,-1]]).T  # narrow band

    kernel_left_mask = torch.tensor([
        [af[2],af[2],af[2],-1,-1,],
        [af[2],af[2],af[2],-1,-1,],
        [af[2],af[2],af[2],-1,-1,],
        [af[2],af[2],af[2],-1,-1,],
        [af[2],af[2],af[2],-1,-1,]]).T  # edge

    kernel_upper_mask = torch.tensor([
        [af[3],af[3],af[3],af[3],af[3]],
        [af[3],af[3],af[3],af[3],af[3]],
        [af[3],af[3],af[3],af[3],af[3]],
        [-1,   -1,   -1,   -1,   -1,],
        [-1,   -1,   -1,   -1,   -1,]]).T  # edge

    kernel_right_mask =  kernel_left_mask.flipud()  # edge
    kernel_lower_mask = kernel_upper_mask.fliplr()  # edge

    state_dict = net.state_dict()
    # numeric weights
    state_dict['conv3.weight'][0,0,:,:] = kernel_vertical[None,None,:,:].to(device)    # increase dimention of kernel
    state_dict['conv3.weight'][1,0,:,:] = kernel_horizontal[None,None,:,:].to(device)   # increase dimention of kernel
    state_dict['conv5.weight'][0,0,:,:] = kernel_left_mask[None,None,:,:].to(device)    # increase dimention of kernel
    state_dict['conv5.weight'][1,0,:,:] = kernel_right_mask[None,None,:,:].to(device)   # increase dimention of kernel
    state_dict['conv5.weight'][2,0,:,:] = kernel_upper_mask[None,None,:,:].to(device)   # increase dimention of kernel
    state_dict['conv5.weight'][3,0,:,:] = kernel_lower_mask[None,None,:,:].to(device)   # increase dimention of kernel

    # bool weights
    state_dict['conv3_bool.weight'][0,0,:,:] = (kernel_vertical[None,None,:,:]   >0)*1.0  # increase dimention of kernel
    state_dict['conv3_bool.weight'][1,0,:,:] = (kernel_horizontal[None,None,:,:] >0)*1.0  # increase dimention of kernel
    state_dict['conv5_bool.weight'][0,0,:,:] = (kernel_left_mask[None,None,:,:]  >0)*1.0  # increase dimention of kernel
    state_dict['conv5_bool.weight'][1,0,:,:] = (kernel_right_mask[None,None,:,:] >0)*1.0  # increase dimention of kernel
    state_dict['conv5_bool.weight'][2,0,:,:] = (kernel_upper_mask[None,None,:,:] >0)*1.0  # increase dimention of kernel
    state_dict['conv5_bool.weight'][3,0,:,:] = (kernel_lower_mask[None,None,:,:] >0)*1.0  # increase dimention of kernel
    
    net.load_state_dict(state_dict)

    return net.to(device)




"""
class RFIconv(nn.Module):

    def __init__(self):
        
        super(RFIconv, self).__init__()

        self.conv3_0 = nn.Conv2d(1,1,kernel_size=(3,3),stride=1,padding=1,bias=False)
        self.conv3_1 = nn.Conv2d(1,1,kernel_size=(3,3),stride=1,padding=1,bias=False)

        self.conv5_0 = nn.Conv2d(1,1,kernel_size=(5,5),stride=1,padding=2,bias=False)
        self.conv5_1 = nn.Conv2d(1,1,kernel_size=(5,5),stride=1,padding=2,bias=False)
        self.conv5_2 = nn.Conv2d(1,1,kernel_size=(5,5),stride=1,padding=2,bias=False)
        self.conv5_3 = nn.Conv2d(1,1,kernel_size=(5,5),stride=1,padding=2,bias=False)
        
        self.conv3_0_bool = nn.Conv2d(1,1,kernel_size=(3,3),stride=1,padding=1,bias=False)
        self.conv3_1_bool = nn.Conv2d(1,1,kernel_size=(3,3),stride=1,padding=1,bias=False)
        
        self.conv5_0_bool = nn.Conv2d(1,1,kernel_size=(5,5),stride=1,padding=2,bias=False)
        self.conv5_1_bool = nn.Conv2d(1,1,kernel_size=(5,5),stride=1,padding=2,bias=False)
        self.conv5_2_bool = nn.Conv2d(1,1,kernel_size=(5,5),stride=1,padding=2,bias=False)
        self.conv5_3_bool = nn.Conv2d(1,1,kernel_size=(5,5),stride=1,padding=2,bias=False)
        
    def forward(self, x):

        dum0 = torch.tensor([0.0])

        c30mask = torch.heaviside(F.conv2d(torch.heaviside(self.conv3_0(x),dum0),torch.ones([1,1,5,5]),stride=1,padding=2),dum0)
        c31mask = torch.heaviside(F.conv2d(torch.heaviside(self.conv3_1(x),dum0),torch.ones([1,1,5,5]),stride=1,padding=2),dum0)

        y = ( self.conv3_0_bool( (torch.heaviside(self.conv3_0(x),dum0))) + self.conv3_1_bool( (torch.heaviside(self.conv3_1(x),dum0)) )
            + self.conv5_1_bool( torch.heaviside(torch.heaviside(self.conv5_0(x),dum0)-c30mask,dum0)) 
            + self.conv5_0_bool( torch.heaviside(torch.heaviside(self.conv5_1(x),dum0)-c30mask,dum0))
            + self.conv5_3_bool( torch.heaviside(torch.heaviside(self.conv5_2(x),dum0)-c31mask,dum0)) 
            + self.conv5_2_bool( torch.heaviside(torch.heaviside(self.conv5_3(x),dum0)-c31mask,dum0)) )>1e-5
        return y

net=Net()

aggressive_factor =[1.6,1.65,0.5,0.5] # (0,4] recommended, the larger the more aggressive
#aggressive_factor = [1.7,1.7] # (0,4] recommended, the larger the more aggressive


def init_RFIconv(net,aggressive_factor = [1.6,1.65,0.5,0.5]):
    af = aggressive_factor
    # for lines
    kernel_vertical = torch.tensor([
        [-1,af[0],-1],
        [-1,af[0],-1],
        [-1,af[0],-1]]).T # transient

    kernel_horizontal = torch.tensor([
        [-1,-1,-1],
        [af[1],af[1],af[1]],
        [-1,-1,-1]]).T  # narrow band

    kernel_left_mask = torch.tensor([
        [af[2],af[2],af[2],-1,-1,],
        [af[2],af[2],af[2],-1,-1,],
        [af[2],af[2],af[2],-1,-1,],
        [af[2],af[2],af[2],-1,-1,],
        [af[2],af[2],af[2],-1,-1,]]).T  # edge

    kernel_upper_mask = torch.tensor([
        [af[3],af[3],af[3],af[3],af[3]],
        [af[3],af[3],af[3],af[3],af[3]],
        [af[3],af[3],af[3],af[3],af[3]],
        [-1,   -1,   -1,   -1,   -1,],
        [-1,   -1,   -1,   -1,   -1,]]).T  # edge

    kernel_right_mask =  kernel_left_mask.flipud()  # edge
    kernel_lower_mask = kernel_upper_mask.fliplr()  # edge

    state_dict = net.state_dict()
    # numeric weights
    state_dict['conv3_0.weight'] = kernel_vertical[None,None,:,:]     # increase dimention of kernel
    state_dict['conv3_1.weight'] = kernel_horizontal[None,None,:,:]   # increase dimention of kernel
    state_dict['conv5_0.weight'] = kernel_left_mask[None,None,:,:]    # increase dimention of kernel
    state_dict['conv5_1.weight'] = kernel_right_mask[None,None,:,:]   # increase dimention of kernel
    state_dict['conv5_2.weight'] = kernel_upper_mask[None,None,:,:]   # increase dimention of kernel
    state_dict['conv5_3.weight'] = kernel_lower_mask[None,None,:,:]   # increase dimention of kernel

    # bool weights
    state_dict['conv3_0_bool.weight'] = (kernel_vertical[None,None,:,:]   >0)*1.0  # increase dimention of kernel
    state_dict['conv3_1_bool.weight'] = (kernel_horizontal[None,None,:,:] >0)*1.0  # increase dimention of kernel
    state_dict['conv5_0_bool.weight'] = (kernel_left_mask[None,None,:,:]  >0)*1.0  # increase dimention of kernel
    state_dict['conv5_1_bool.weight'] = (kernel_right_mask[None,None,:,:] >0)*1.0  # increase dimention of kernel
    state_dict['conv5_2_bool.weight'] = (kernel_upper_mask[None,None,:,:] >0)*1.0  # increase dimention of kernel
    state_dict['conv5_3_bool.weight'] = (kernel_lower_mask[None,None,:,:] >0)*1.0  # increase dimention of kernel
    
    net.load_state_dict(state_dict)

    return net

#net  = init_net(net,aggressive_factor)
"""