import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchinfo import summary

# #--------------------------------------------------------                
# # model 00
# #--------------------------------------------------------
class Model_v00(nn.Module):
    def __init__(self, dropout=True):
        super().__init__()
        self.dropout = dropout
        if self.dropout:
            self.name = "Model_v00"
        else:
            self.name = "Model_v00_No-DO"
            
        # convolution layers
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten()
        )

        
        # Fully connected layers
        self.head = nn.Sequential(
            nn.Linear(in_features=64*52*52, out_features=1024), 
            nn.ReLU(inplace=True),
            
            nn.Linear(in_features=1024, out_features=3)
            
        )
    
    def forward(self, x):
        
        # apply feature extractor
        x = self.body(x)
        # apply classification head
        x = self.head(x)
        
        return x


# #--------------------------------------------------------                
# # model 01
# #--------------------------------------------------------
class Model_v01(nn.Module):
    def __init__(self, dropout=True):
        super().__init__()
        self.dropout = dropout
        if self.dropout:
            self.name = "Model_v01"
        else:
            self.name = "Model_v01_No-DO"
            
        # convolution layers
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten()
        )
        
        # Fully connected layers
        self.head = nn.Sequential(
            nn.Linear(in_features=128*24*24, out_features=1024), 
            nn.ReLU(inplace=True),
            
            nn.Linear(in_features=1024, out_features=3)
            
        )
    
    def forward(self, x):
        
        # apply feature extractor
        x = self.body(x)
        # apply classification head
        x = self.head(x)
        
        return x

# #--------------------------------------------------------                
# # model 02
# #--------------------------------------------------------
class Model_v02(nn.Module):
    def __init__(self, dropout=True):
        super().__init__()
        self.dropout = dropout
        if self.dropout:
            self.name = "Model_v02"
        else:
            self.name = "Model_v02_No-DO"
            
        # convolution layers
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten()
        )
        
        # Fully connected layers
        self.head = nn.Sequential(
            nn.Linear(in_features=256*24*24, out_features=1024), 
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=3)
        )

    
    def forward(self, x):        
        # apply feature extractor
        x = self.body(x)
        # flatten the output of conv layers
        # dimension should be batch_size * number_of weight_in_last conv_layer
        # x = x.view(x.size()[0], -1)
        # apply classification head
        x = self.head(x)
        
        return x


# #--------------------------------------------------------                
# # model 03 (model 2 + extra layer in trunk)
# #--------------------------------------------------------
class Model_v031(nn.Module):
    def __init__(self, dropout=True):
        super().__init__()
        self.dropout = dropout
        if self.dropout:
            self.name = "Model_v031"
        else:
            self.name = "Model_v031_No-DO"
            
        # convolution layers
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Flatten()
        )
        
        # Fully connected layers
        self.head = nn.Sequential(
            nn.Linear(in_features=512*10*10, out_features=1024), 
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=3)
        )
    
    def forward(self, x):
        # apply feature extractor
        x = self.body(x)
        # apply classification head
        x = self.head(x)
        return x



##--------------------------------------------------------                
## model 031 (model 30 + extra layer in head)
##--------------------------------------------------------
class Model_v031(nn.Module):
    def __init__(self, dropout=True):
        super().__init__()
        self.dropout = dropout
        if self.dropout:
            self.name = "Model_v031"
        else:
            self.name = "Model_v031_No-DO"
            
        # convolution layers
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Flatten()
        )
        
        # Fully connected layers
        self.head = nn.Sequential(
            nn.Linear(in_features=512*10*10, out_features=2048), 
            nn.ReLU(inplace=True),
            
            nn.Linear(in_features=2048, out_features=1024), 
            nn.ReLU(inplace=True),
            
            nn.Linear(in_features=1024, out_features=3)
        )
    
    def forward(self, x):
        # apply feature extractor
        x = self.body(x)
        # apply classification head
        x = self.head(x)
        return x


##--------------------------------------------------------------------------
## model 04 4 Trunk layers, 3 head layers (model 31 + Batch Normalization)
##--------------------------------------------------------------------------

class Model_v04(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p
        self.name = "Model_v04"
            
        # convolution layers
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Flatten()
        )
        
        # Fully connected layers
        self.head = nn.Sequential(
            nn.Linear(in_features=512*10*10, out_features=2048), 
            nn.ReLU(inplace=True),
            
            nn.Linear(in_features=2048, out_features=1024), 
            nn.ReLU(inplace=True),
            
            nn.Linear(in_features=1024, out_features=3)
        )
    
    def forward(self, x):
        # apply feature extractor
        x = self.body(x)
        # apply classification head
        x = self.head(x)
        return x



##--------------------------------------------------------
## model 04 4 Trunk layers, 2 head layers (model 31 + Batch Normalization)
##--------------------------------------------------------
class Model_v42(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p
        self.name = "Model_v42"
            
        # convolution layers
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Flatten()
        )
        
        # Fully connected layers
        self.head = nn.Sequential(
            nn.Linear(in_features=512*10*10, out_features=2048), 
            nn.ReLU(inplace=True),
            
            nn.Linear(in_features=2048, out_features=3)
        )
    
    def forward(self, x):
        # apply feature extractor
        x = self.body(x)
        # apply classification head
        x = self.head(x)
        return x



##--------------------------------------------------------
## model 5 - 5 trunk layers, 3 head laeyrs
##--------------------------------------------------------
class Model_v05(nn.Module):
    def __init__(self, dropout=True):
        super().__init__()
        self.dropout = dropout
        self.name = "Model_v05"
 
        # convolution layers
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=5),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),            

            nn.Flatten()
        )
        
        # Fully connected layers
        self.head = nn.Sequential(
            # nn.Linear(in_features=512*10*10, out_features=2048), 
            nn.Linear(in_features=1024*3*3, out_features=2048), 
            nn.ReLU(inplace=True),
            
            nn.Linear(in_features=2048, out_features=1024), 
            nn.ReLU(inplace=True),
            
            nn.Linear(in_features=1024, out_features=3)
        )
    
    def forward(self, x):
        # apply feature extractor
        x = self.body(x)
        # flatten the output of conv layers
        # dimension should be batch_size * number_of weight_in_last conv_layer
        # x = x.view(x.size()[0], -1)
        # apply classification head
        x = self.head(x)
        return x        



##--------------------------------------------------------
## model 5 - 5 trunk layers, 
##            3 -> 64 -> 128 -> 256 -> 512 -> 1024 -> 2048
##           3 head layers
##            2048 * 4 -> 2048 -> 1024 -> 3
##--------------------------------------------------------
class Model_v051(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.name = "Model_v51"
            
        # convolution layers
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding = 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding = 2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding = 2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, padding = 2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=5, padding = 2),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),            

            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),            
            
            nn.Flatten()
        )
        
        # Fully connected layers
        self.head = nn.Sequential(
            # nn.Linear(in_features=512*10*10, out_features=2048), 
            nn.Linear(in_features=2048*2*2, out_features=2048), 
            nn.ReLU(inplace=True),
            
            nn.Linear(in_features=2048, out_features=1024), 
            nn.ReLU(inplace=True),
            
            nn.Linear(in_features=1024, out_features=3)
        )
    
    def forward(self, x):
        # apply feature extractor
        x = self.body(x)
        # apply classification head
        x = self.head(x)
        return x        





##--------------------------------------------------------
## model 52 - 6 trunk layers : 
##             3 -> 64 -> 256 -> 512 -> 1024 -> 1024 ->2048
##
##           Head layers
##            2048 * 4 -> 2048 -> 1024 -> dropout -> 3
##--------------------------------------------------------
class Model_v052(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.name = "Model_v52"
        self.p = p
        
        # convolution layers
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding = 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=5, padding = 2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, padding = 2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=5, padding = 2),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=5, padding = 2),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),            

            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),            
            nn.Flatten()
        )
        
        # Fully connected layers
        self.head = nn.Sequential(
            # nn.Linear(in_features=512*10*10, out_features=2048), 
            nn.Linear(in_features=2048*2*2, out_features=2048), 
            nn.ReLU(inplace=True),
            
            nn.Linear(in_features=2048, out_features=1024), 
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=self.p),
            
            nn.Linear(in_features=1024, out_features=3)
        )
    
    def forward(self, x):
        # apply feature extractor
        x = self.body(x)
        # apply classification head
        x = self.head(x)
        return x        


##--------------------------------------------------------
## model 53 - 5 trunk layers, 3 head laeyrs
##--------------------------------------------------------
class Model_v053(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.name = "Model_v53"
        self.p = p 
        l1_out = 256
        # convolution layers
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=256, kernel_size=7, padding = 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),
            
            nn.Conv2d(in_channels=l1_out, out_channels=512, kernel_size=5, padding = 2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=5, padding = 2),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),

            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=5, padding = 2),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=5, padding = 2),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                 
            
            nn.Flatten()
        )
        
        # Fully connected layers
        self.head = nn.Sequential(
            # nn.Linear(in_features=512*10*10, out_features=2048), 
            nn.Linear(in_features=1024*2*2, out_features=2048), 
            nn.ReLU(inplace=True),
            
            nn.Linear(in_features=2048, out_features=1024), 
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=self.p),
            nn.Linear(in_features=1024, out_features=3)
        )
    
    def forward(self, x):
        # apply feature extractor
        x = self.body(x)
        # apply classification head
        x = self.head(x)
        return x      

        
##--------------------------------------------------------
## model 6 - 5 trunk layers - : 
##           3 -> 64 -> 128 -> -> 256 -> 512 -> 1024 , 3 head layers + 1 dropout in head
##--------------------------------------------------------
 
class Model_v06(nn.Module):
    def __init__(self, p = 0.2):
        super().__init__()
        self.name = "Model_v06"
        self.p = p
        
        # convolution layers
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=5),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),            

            nn.Flatten()
        )
        
        # Fully connected layers
        self.head = nn.Sequential(
            # nn.Linear(in_features=512*10*10, out_features=2048), 
            nn.Linear(in_features=1024*3*3, out_features=2048), 
            nn.ReLU(inplace=True),
            
            nn.Linear(in_features=2048, out_features=1024), 
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=self.p),
            nn.Linear(in_features=1024, out_features=3)
        )
    
    def forward(self, x):
        # apply feature extractor
        x = self.body(x)
        # flatten the output of conv layers
        # dimension should be batch_size * number_of weight_in_last conv_layer
        # x = x.view(x.size()[0], -1)
        # apply classification head
        x = self.head(x)
        return x        


##--------------------------------------------------------
## model 61 - 6 trunk layers : 
##             3 -> 64 -> 128 -> -> 256 -> 512 -> 1024 ->2048   (Model 6 + 1 additional trunk layer) 
##   3 head layers (one sroput)
##--------------------------------------------------------
class Model_v61(nn.Module):
    def __init__(self, p = 0.2):
        super().__init__()
        self.name = "Model_v61"
        self.p = p
        
        # convolution layers
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding= 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding = 2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding = 2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, padding =2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=5, padding =2),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),            
            
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=5, padding = 1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),            
            
            nn.Flatten()
        )
        
        # Fully connected layers
        self.head = nn.Sequential(
            nn.Linear(in_features=2048*2*2, out_features=2048), 
            nn.ReLU(inplace=True),
           
            nn.Linear(in_features=2048, out_features=1024), 
            nn.ReLU(inplace=True),
           
            nn.Dropout(p=self.p),
            nn.Linear(in_features=1024, out_features=3)
        )
    
    def forward(self, x):
        # apply feature extractor
        x = self.body(x)
        # apply classification head
        x = self.head(x)
        return x         




##--------------------------------------------------------
## model 61 (Model 61 with no batch layers in L1, L2) 
##   3 head layers (one sroput)
##--------------------------------------------------------
class Model_v61a(nn.Module):
    def __init__(self, p = 0.2):
        super().__init__()
        self.name = "Model_v61a"
        self.p = p
        
        # convolution layers
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding= 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding = 2),
            # nn.BatchNorm2d(128),     <-- difference with v61
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding = 2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, padding =2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=5, padding =2),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),            
            
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=5, padding = 1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),            
            
            nn.Flatten()
        )
        
        # Fully connected layers
        self.head = nn.Sequential(
            nn.Linear(in_features=2048*2*2, out_features=2048), 
            nn.ReLU(inplace=True),
           
            nn.Linear(in_features=2048, out_features=1024), 
            nn.ReLU(inplace=True),
           
            nn.Dropout(p=self.p),
            nn.Linear(in_features=1024, out_features=3)
        )
    
    def forward(self, x):
        # apply feature extractor
        x = self.body(x)
        # apply classification head
        x = self.head(x)
        return x         


##--------------------------------------------------------
## model 61b (Model 61 with batch layers in L1 and other layers) 
##           Batch Norm in L1-6
##           3 head layers (one sroput)
##--------------------------------------------------------
class Model_v61b(nn.Module):
    def __init__(self, p = 0.2):
        super().__init__()
        self.name = "Model_v61b"
        self.p = p
        
        # convolution layers
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding= 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding = 2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding = 2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, padding =2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=5, padding =2),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),            
            
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=5, padding = 1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),            
            
            nn.Flatten()
        )
        
        # Fully connected layers
        self.head = nn.Sequential(
            nn.Linear(in_features=2048*2*2, out_features=2048), 
            nn.ReLU(inplace=True),
           
            nn.Linear(in_features=2048, out_features=1024), 
            nn.ReLU(inplace=True),
           
            nn.Dropout(p=self.p),
            nn.Linear(in_features=1024, out_features=3)
        )
    
    def forward(self, x):
        # apply feature extractor
        x = self.body(x)
        # apply classification head
        x = self.head(x)
        return x         





##--------------------------------------------------------
## model 61c - Model 61 
##           Batch Norm in L1-6
##           Drop out in L1 and L2
##--------------------------------------------------------
class Model_v61c(nn.Module):
    def __init__(self, p = 0.2):
        super().__init__()
        self.name = "Model_v61c"
        self.p = p
        
        # convolution layers
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding= 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=self.p),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding = 2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=self.p),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding = 2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, padding =2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=5, padding =2),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),            
            
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=5, padding = 1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),            
            
            nn.Flatten()
        )
        
        # Fully connected layers
        self.head = nn.Sequential(
            nn.Linear(in_features=2048*2*2, out_features=2048), 
            nn.ReLU(inplace=True),
           
            nn.Linear(in_features=2048, out_features=1024), 
            nn.ReLU(inplace=True),
           
            nn.Linear(in_features=1024, out_features=3)
        )
    
    def forward(self, x):
        # apply feature extractor
        x = self.body(x)
        # apply classification head
        x = self.head(x)
        return x         

        
##--------------------------------------------------------
## model 62 (Model 61 + with smaller kernel sizes) - 6 trunk layers : 
##             3 -> 64 -> 128 -> -> 256 -> 512 -> 1024 -> 2048   (Model 6 + 1 additional trunk layer) 
##   3 head layers (one sroput)
##--------------------------------------------------------
class Model_v62(nn.Module):
    def __init__(self, p = 0.2):
        super().__init__()
        self.name = "Model_v62"
        self.p = p
        k3 = 3
        k5 = 5
        # convolution layers
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=k5, padding= 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=k3, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=k3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=k3, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=k3, padding = 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),            
            
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=k3, padding = 1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),            
            
            nn.Flatten()
        )
        
        # Fully connected layers
        self.head = nn.Sequential(
            nn.Linear(in_features=2048*3*3, out_features=2048), 
            nn.ReLU(inplace=True),
           
            nn.Linear(in_features=2048, out_features=1024), 
            nn.ReLU(inplace=True),
           
            nn.Dropout(p=self.p),
            nn.Linear(in_features=1024, out_features=3)
        )
    
    def forward(self, x):
        # apply feature extractor
        x = self.body(x)
        # apply classification head
        x = self.head(x)
        return x         


##--------------------------------------------------------
## model 62a (Model 62 + with no batch norm in L1, L2) 
##   3 head layers (one sroput)
##--------------------------------------------------------
class Model_v62a(nn.Module):
    def __init__(self, p = 0.2):
        super().__init__()
        self.name = "Model_v62a"
        self.p = p
        k3 = 3
        k5 = 5
        # convolution layers
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=k5, padding= 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=k3, padding = 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=k3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=k3, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=k3, padding = 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),            
            
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=k3, padding = 1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),            
            
            nn.Flatten()
        )
        
        # Fully connected layers
        self.head = nn.Sequential(
            nn.Linear(in_features=2048*3*3, out_features=2048), 
            nn.ReLU(inplace=True),
           
            nn.Linear(in_features=2048, out_features=1024), 
            nn.ReLU(inplace=True),
           
            nn.Dropout(p=self.p),
            nn.Linear(in_features=1024, out_features=3)
        )
    
    def forward(self, x):
        # apply feature extractor
        x = self.body(x)
        # apply classification head
        x = self.head(x)
        return x         


##--------------------------------------------------------
## model 62a (Model 62 + batch norm in all trunk layers) 
##   3 head layers (one sroput)
##--------------------------------------------------------
class Model_v62b(nn.Module):
    def __init__(self, p = 0.2):
        super().__init__()
        self.name = "Model_v62a"
        self.p = p
        k3 = 3
        k5 = 5
        # convolution layers
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=k5, padding= 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=k3, padding = 1),
            nn.BatchNorm2d(128),            
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=k3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=k3, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=k3, padding = 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),            
            
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=k3, padding = 1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),            
            
            nn.Flatten()
        )
        
        # Fully connected layers
        self.head = nn.Sequential(
            nn.Linear(in_features=2048*3*3, out_features=2048), 
            nn.ReLU(inplace=True),
           
            nn.Linear(in_features=2048, out_features=1024), 
            nn.ReLU(inplace=True),
           
            nn.Dropout(p=self.p),
            nn.Linear(in_features=1024, out_features=3)
        )
    
    def forward(self, x):
        # apply feature extractor
        x = self.body(x)
        # apply classification head
        x = self.head(x)
        return x         


##--------------------------------------------------------
## model 7 - 7 trunk layers : 
##            3 -> 64 -> 128 -> 256 -> 512 -> 1024 -> 2048   (Model 6 + 1 additional trunk layer) 
##--------------------------------------------------------

class Model_v07(nn.Module):
    def __init__(self, p = 0.2):
        super().__init__()
        self.name = "Model_v07"
        self.p = p
        
        # convolution layers
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding = 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding = 2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding = 2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, padding = 2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=5, padding = 2),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),            

            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=5, padding = 1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),            
            nn.Dropout(p=self.p),

            nn.Conv2d(in_channels=2048, out_channels=4096, kernel_size=k3, padding = 1),
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, ),            
                        
            nn.Flatten()
        )
        
        # Fully connected layers
        self.head = nn.Sequential(
            nn.Linear(in_features=2048*2*2, out_features=2048), 
            nn.ReLU(inplace=True),
            
            nn.Linear(in_features=2048, out_features=1024), 
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=self.p),
            nn.Linear(in_features=1024, out_features=3)
        )
    
    def forward(self, x):
        # apply feature extractor
        x = self.body(x)
        # flatten the output of conv layers
        # dimension should be batch_size * number_of weight_in_last conv_layer
        # x = x.view(x.size()[0], -1)
        # apply classification head
        x = self.head(x)
        return x        
 



##--------------------------------------------------------
## model 71 - 7 trunk layers : 
##            3 -> 64 -> 128 -> 256 -> 512 -> 1024 -> 2048   (Model 6 + 1 additional trunk layer) 
##--------------------------------------------------------

class Model_v71(nn.Module):
    def __init__(self, p = 0.2):
        super().__init__()
        self.name = "Model_v71"
        self.p = p
        
        # convolution layers
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding = 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding = 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),            

            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, padding = 1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),            

            nn.Conv2d(in_channels=2048, out_channels=4096, kernel_size=3, padding = 1),
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, ),            
                        
            nn.Flatten()
        )
        
        # Fully connected layers
        self.head = nn.Sequential(
            nn.Linear(in_features=4096 * 1, out_features=2048), 
            nn.ReLU(inplace=True),
            
            nn.Linear(in_features=2048, out_features=1024), 
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=self.p),
            nn.Linear(in_features=1024, out_features=3)
        )
    
    def forward(self, x):
        # apply feature extractor
        x = self.body(x)
        # flatten the output of conv layers
        # dimension should be batch_size * number_of weight_in_last conv_layer
        # x = x.view(x.size()[0], -1)
        # apply classification head
        x = self.head(x)
        return x        
 

##--------------------------------------------------------
## model 72b - 7 trunk layers : 
##            3 -> 64 -> 128 -> 256 -> 512 -> 1024 -> 2048   (Model 6 + 1 additional trunk layer) 
##--------------------------------------------------------
class Model_v71b(nn.Module):
    def __init__(self, p = 0.2):
        super().__init__()
        self.name = "Model_v71b"
        self.p = p
        k3 = 3
        k5 = 5
        k7 = 7 
        # convolution layers
        
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=k5, padding= 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=k3, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=k3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=k3, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=k3, padding = 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),            
            
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=k3, padding = 1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),            

            nn.Conv2d(in_channels=2048, out_channels=4096, kernel_size=k3, padding = 1),
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, ),            
                        
            nn.Flatten()
        )
        
        # Fully connected layers
        self.head = nn.Sequential(
            nn.Linear(in_features=4096*1, out_features=2048), 
            nn.ReLU(inplace=True),
           
            nn.Linear(in_features=2048, out_features=1024), 
            nn.ReLU(inplace=True),
           
            nn.Dropout(p=self.p),
            nn.Linear(in_features=1024, out_features=3)
        )
    
    def forward(self, x):
        # apply feature extractor
        x = self.body(x)
        # apply classification head
        x = self.head(x)
        return x         



##--------------------------------------------------------
## model 61 (Model 6 + 1 additional trunk layer) 
##   3 head layers (one sroput)
##--------------------------------------------------------
class Model_v71c(nn.Module):
    def __init__(self, p = 0.2):
        super().__init__()
        self.name = "Model_v71c"
        self.p = p
        k3 = 3
        k5 = 5
        k7 = 7 
        # convolution layers
        
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=k5, padding= 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=self.p),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=k3, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=self.p),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=k3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=k3, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=k3, padding = 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),            
            
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=k3, padding = 1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),            

            nn.Conv2d(in_channels=2048, out_channels=4096, kernel_size=k3, padding = 1),
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, ),            
                        
            nn.Flatten()
        )
        
        # Fully connected layers
        self.head = nn.Sequential(
            nn.Linear(in_features=4096*1, out_features=2048), 
            nn.ReLU(inplace=True),
           
            nn.Linear(in_features=2048, out_features=1024), 
            nn.ReLU(inplace=True),
           
            # nn.Dropout(p=self.p),
            nn.Linear(in_features=1024, out_features=3)
        )
    
    def forward(self, x):
        # apply feature extractor
        x = self.body(x)
        # apply classification head
        x = self.head(x)
        return x         
 

##--------------------------------------------------------
## model 81b - 7 trunk layers : 
##            3 -> 256-> 256 -> 512 -> 512 -> 2048 -> 2048 -> 4096   (Model 6 + 1 additional trunk layer) 
##--------------------------------------------------------
class Model_v81b(nn.Module):
    def __init__(self, p = 0.2):
        super().__init__()
        self.name = "Model_v81b"
        self.p = p
        k3 = 3
        k5 = 5
        k7 = 7
        l1_out = 256
        l2_out = 512
        l3_out = 2048
        l4_out = 4096
        # convolution layers
        
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=l1_out, kernel_size=k5, padding= 2),
            nn.BatchNorm2d(l1_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=l1_out, out_channels=l1_out, kernel_size=k3, padding = 1),
            nn.BatchNorm2d(l1_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=l1_out, out_channels=l2_out, kernel_size=k3, padding = 1),
            nn.BatchNorm2d(l2_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=l2_out, out_channels=l2_out, kernel_size=k3, padding = 1),
            nn.BatchNorm2d(l2_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=l2_out, out_channels=l3_out, kernel_size=k3, padding = 1),
            nn.BatchNorm2d(l3_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),            
            
            nn.Conv2d(in_channels=l3_out, out_channels=l3_out, kernel_size=k3, padding = 1),
            nn.BatchNorm2d(l3_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),            

            nn.Conv2d(in_channels=l3_out, out_channels=l4_out, kernel_size=k3, padding = 1),
            nn.BatchNorm2d(l4_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, ),            
                        
            nn.Flatten()
        )
        
        # Fully connected layers
        self.head = nn.Sequential(
            nn.Linear(in_features=l4_out*1, out_features=2048), 
            nn.ReLU(inplace=True),
           
            nn.Linear(in_features=2048, out_features=1024), 
            nn.ReLU(inplace=True),
           
            nn.Dropout(p=self.p),
            nn.Linear(in_features=1024, out_features=3)
        )
    
    def forward(self, x):
        # apply feature extractor
        x = self.body(x)
        # apply classification head
        x = self.head(x)
        return x         



##--------------------------------------------------------
## model 
##--------------------------------------------------------

class Model_vxx(nn.Module):
    def __init__(self, p = 0.2):
        super().__init__()
        self.name = "Model_vxx"
        self.p = p
        
        # convolution layers
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding= 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, padding= 2),
            nn.ReLU(inplace=True),            
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, padding = 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5 ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=5, ),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),            
            
            nn.Flatten()
        )
        
        # Fully connected layers
        self.head = nn.Sequential(
            nn.Linear(in_features=1024*4*4, out_features=2048), 
            nn.ReLU(inplace=True),
            nn.Linear(in_features=2048, out_features=1024), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.p),
            nn.Linear(in_features=1024, out_features=3)
        )
    
    def forward(self, x):
        # apply feature extractor
        x = self.body(x)
        # apply classification head
        x = self.head(x)
        return x         


##--------------------------------------------------------
## model 
##--------------------------------------------------------
class MyModel_vxx(nn.Module):
    def __init__(self, dropout = True):
        super().__init__()
        self.dropout = dropout
        if self.dropout:
            self.name = "MyModel_v40"
        else:
            self.name = "MyModel_v40_No-DO"
        pad1 = 1
        dil = 1
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=3 , out_channels=64, kernel_size=3, dilation=dil, padding = pad),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, dilation=dil, padding = pad),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, dilation=dil, padding = pad),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, dilation=dil, padding = pad),
            nn.BatchNorm2d(256), 
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, dilation=dil, padding = pad),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, dilation=dil, padding = pad),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # nn.Dropout(p=0.2) if self.dropout else nn.Identity(), 

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding = pad),
            nn.BatchNorm2d(512), 
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding = pad),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding = pad),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # nn.Dropout(p=0.2) if self.dropout else nn.Identity(), 

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding = pad),
            nn.BatchNorm2d(1024), 
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding = pad),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding = pad),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # nn.Dropout(p=0.2) if self.dropout else nn.Identity(), 

            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(9216, 4096),
            nn.ReLU(),   
            nn.Linear(4096, 2048),
            nn.ReLU(),   

            # nn.Dropout(p=0.2) if self.dropout else nn.Identity(), 

            nn.Linear(2048, 3)
        )

    def forward(self, x):
        ### BEGIN SOLUTION
        x = self.body(x)
        x = self.head(x)
        ### END SOLUTION
        return x

##--------------------------------------------------------
## model 
##--------------------------------------------------------
 


 


if __name__ == "__main__":
    my_model = MyModel_v18()

    print(my_model)
    col_names = ("input_size",
                "output_size",
                "num_params",
                # "params_percent",
                "kernel_size",
                "mult_adds",
                "trainable" )
    batch_size = 4
    input_size = (3, 32, 32)
    
    # summary(my_model.body, input_size = (batch_size,3, 32, 32), col_names=col_names)
    print(summary(my_model, input_size=(batch_size, 3, 32, 32), col_names=col_names))
