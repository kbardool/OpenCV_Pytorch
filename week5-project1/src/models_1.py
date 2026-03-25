import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchinfo import summary

##--------------------------------------------------------
## model 00
##   Trunk:  Batch Norm in L1-6
##    Head:  3 head layers (one sroput)
##--------------------------------------------------------
class MyModel_v00(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "MyModel_v00"

        self.body = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2),     
            
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2), 
            
            nn.Conv2d(64, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            
            # nn.Conv2d(32, 32, 3),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            # nn.MaxPool2d(2, 2), 
            
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(5408, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256), 
            nn.ReLU(),           
            nn.Linear(256, 10)
        )

    def forward(self, x):
        ### BEGIN SOLUTION
        x = self.body(x)
        x = self.head(x)
        ### END SOLUTION
        return x

##--------------------------------------------------------
## model 12
##   Trunk:  
##    Head:  
##--------------------------------------------------------
class MyModel_v12(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "MyModel_v12"
        
        self.body = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),     
            
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512), 
            nn.ReLU(),
            nn.Linear(512, 256), 
            nn.ReLU(),          
            nn.Linear(256, 10)
        )

    def forward(self, x):
        ### BEGIN SOLUTION
        x = self.body(x)
        x = self.head(x)
        ### END SOLUTION
        return x


##--------------------------------------------------------
## model 13
##   Trunk:  
##    Head:  
##--------------------------------------------------------
class MyModel_v13(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "MyModel_v13"

        self.body = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),     
            
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Flatten()
        )
        self.head = nn.Sequential(
            # nn.Linear(1024, 1024),
            # nn.ReLU(),
            nn.Linear(1024, 512), 
            nn.ReLU(),
            nn.Linear(512, 256), 
            nn.ReLU(),          
            nn.Linear(256, 10)
        )

    def forward(self, x):
        ### BEGIN SOLUTION
        x = self.body(x)
        x = self.head(x)
        ### END SOLUTION
        return x


##--------------------------------------------------------
## model 14
##   Trunk:  
##    Head:  
##--------------------------------------------------------
class MyModel_v14(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "MyModel_v14"

        self.body = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),     
            
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(1024, 256), 
            nn.ReLU(),     
            nn.Linear(256, 10)
        )

    def forward(self, x):
        ### BEGIN SOLUTION
        x = self.body(x)
        x = self.head(x)
        ### END SOLUTION
        return x

##--------------------------------------------------------
## model 15
##   Trunk:  
##    Head:  
##--------------------------------------------------------
class MyModel_v15(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "MyModel_v15"
        
        self.body = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),     
            
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(2304, 256), 
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        ### BEGIN SOLUTION
        x = self.body(x)
        x = self.head(x)
        ### END SOLUTION
        return x

##--------------------------------------------------------
## model 16
##   Trunk:  
##    Head:  
##--------------------------------------------------------
class MyModel_v16(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "MyModel_v16"        

        self.body = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),     
            
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(2304, 512), 
            nn.ReLU(),
            # nn.Linear(512, 256), 
            # nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        ### BEGIN SOLUTION
        x = self.body(x)
        x = self.head(x)
        ### END SOLUTION
        return x        

##--------------------------------------------------------
## model 17
##   Trunk:  
##    Head:  
##--------------------------------------------------------
class MyModel_v17(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "MyModel_v17"        

        self.body = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2),     
            
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(2304, 256), 
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        ### BEGIN SOLUTION
        x = self.body(x)
        x = self.head(x)
        ### END SOLUTION
        return x        


##--------------------------------------------------------
## model 18
##   Trunk:  
##    Head:  
##--------------------------------------------------------
class MyModel_v18(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "MyModel_v18"
        
        self.body = nn.Sequential(
            nn.Conv2d(3, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
             
            nn.Conv2d(32, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            
            nn.Conv2d(64, 128, 5),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),   
            nn.Linear(512, 10)
        )

    def forward(self, x):
        ### BEGIN SOLUTION
        x = self.body(x)
        x = self.head(x)
        ### END SOLUTION
        return x


##--------------------------------------------------------
## model 19
##   Trunk:  
##    Head:  
##--------------------------------------------------------
class MyModel_v19(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "MyModel_v19"
        
        self.body = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),     
            
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2), 
            
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),   
            nn.Linear(512, 256),
            nn.ReLU(),   
            nn.Linear(256, 10)
        )

    def forward(self, x):
        ### BEGIN SOLUTION
        x = self.body(x)
        x = self.head(x)
        ### END SOLUTION
        return x


##--------------------------------------------------------
## model 20
##   Trunk:  
##    Head:  
##--------------------------------------------------------
class MyModel_v20(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "MyModel_v20"
        
        self.body = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),     
            
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2), 
            
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),   
            nn.Linear(1024, 256),
            nn.ReLU(),   
            nn.Linear(256, 10)
        )

    def forward(self, x):
        ### BEGIN SOLUTION
        x = self.body(x)
        # x = x.view(x.shape[0], -1)
        x = self.head(x)
        ### END SOLUTION
        return x


##--------------------------------------------------------
## model 21
##   Trunk:  
##    Head:  
##--------------------------------------------------------
class MyModel_v21(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "MyModel_v21"
        
        self.body = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),     
            
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2), 
            
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),   
            nn.Linear(1024, 512),
            nn.ReLU(),   
            nn.Linear(512, 256),
            nn.ReLU(),   
            nn.Linear(256, 10)
        )

    def forward(self, x):
        ### BEGIN SOLUTION
        x = self.body(x)
        # x = x.view(x.shape[0], -1)
        x = self.head(x)
        ### END SOLUTION
        return x



##--------------------------------------------------------
## model 22
##   Trunk:  
##    Head:  
##--------------------------------------------------------
class MyModel_v22(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "MyModel_v22"
        
        self.body = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),     
            
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2), 
            
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),   
            nn.Linear(1024, 512),
            nn.ReLU(),   
            nn.Linear(512, 512),
            nn.ReLU(),   
            nn.Linear(512, 256),
            nn.ReLU(),   
            nn.Linear(256, 10)
        )

    def forward(self, x):
        ### BEGIN SOLUTION
        x = self.body(x)
        x = self.head(x)
        ### END SOLUTION
        return x



##--------------------------------------------------------
## model 23
##   Trunk:  
##    Head:  
##--------------------------------------------------------
class MyModel_v23(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "MyModel_v23"
        
        self.body = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            
            nn.Conv2d(16, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2), 
            
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),   
            nn.Linear(256, 10)
        )

    def forward(self, x):
        ### BEGIN SOLUTION
        x = self.body(x)
        x = self.head(x)
        ### END SOLUTION
        return x


##--------------------------------------------------------
## model 30
##   Trunk:  
##    Head:  
##--------------------------------------------------------
class MyModel_v30(nn.Module):
    def __init__(self, dropout = True):
        super().__init__()
        self.dropout = dropout
        if self.dropout:
            self.name = "MyModel_v30"
        else:
            self.name = "MyModel_v30_No-DO"
        
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=3 , out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), 
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Dropout(p=0.2) if self.dropout else nn.Identity(), 
            
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), 
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Dropout(p=0.2) if self.dropout else nn.Identity(), 
            
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(4096 * 2, 4096 * 2),
            nn.ReLU(),   
            nn.Linear(4096 * 2, 2048 * 2),
            nn.ReLU(),   

            nn.Dropout(p=0.2) if self.dropout else nn.Identity(), 

            nn.Linear(2048 * 2, 10)
        )

    def forward(self, x):
        ### BEGIN SOLUTION
        x = self.body(x)
        x = self.head(x)
        ### END SOLUTION
        return x

##--------------------------------------------------------
## model 31
##   Trunk:  
##    Head:  
##--------------------------------------------------------
class MyModel_v31(nn.Module):
    def __init__(self, dropout = True):
        super().__init__()
        self.dropout = dropout
        if self.dropout:
            self.name = "MyModel_v301"
        else:
            self.name = "MyModel_v31_No-DO"
        
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=3 , out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), 
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2) if self.dropout else nn.Identity(), 
            
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), 
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2) if self.dropout else nn.Identity(), 
            
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(4096 * 2, 4096 * 2),
            nn.ReLU(),   
            nn.Linear(4096 * 2, 2048 * 2),
            nn.ReLU(),   
            nn.Dropout(p=0.2) if self.dropout else nn.Identity(), 
            nn.Linear(2048 * 2, 10)
        )

    def forward(self, x):
        ### BEGIN SOLUTION
        x = self.body(x)
        x = self.head(x)
        ### END SOLUTION
        return x



##--------------------------------------------------------
## model 32
##   Trunk:  
##    Head:  
##--------------------------------------------------------
class MyModel_v32(nn.Module):
    def __init__(self, dropout = True):
        super().__init__()
        self.dropout = dropout
        if self.dropout:
            self.name = "MyModel_v32"
        else:
            self.name = "MyModel_v32_No-DO"
        
        
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=3 , out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), 
            nn.ReLU(),
            # nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2) if self.dropout else nn.Identity(), 
            
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), 
            nn.ReLU(),
            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2) if self.dropout else nn.Identity(), 
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(4096 * 2, 4096 * 2),
            nn.ReLU(),   
            nn.Linear(4096 * 2, 2048 * 2),
            nn.ReLU(),   
            nn.Dropout(p=0.2) if self.dropout else nn.Identity(), 
            nn.Linear(2048 * 2, 10)
        )

    def forward(self, x):
        ### BEGIN SOLUTION
        x = self.body(x)
        x = self.head(x)
        ### END SOLUTION
        return x


##--------------------------------------------------------
## model 33
##   Trunk:  
##    Head:  
##--------------------------------------------------------
class MyModel_v33(nn.Module):
    def __init__(self, dropout = True):
        super().__init__()
        self.dropout = dropout
        if self.dropout:
            self.name = "MyModel_v33"
        else:
            self.name = "MyModel_v33_No-DO"
        
        
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=3 , out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), 
            nn.ReLU(),
            # nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2) if self.dropout else nn.Identity(), 
            
            # nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512), 
            # nn.ReLU(),
            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2) if self.dropout else nn.Identity(), 
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(256 * 4* 4, 256 * 4* 4),
            nn.ReLU(),   
            nn.Linear(256 * 4 * 4, 2048 * 2),
            nn.ReLU(),   
            nn.Dropout(p=0.2) if self.dropout else nn.Identity(), 
            nn.Linear(2048 * 2, 10)
        )

    def forward(self, x):
        ### BEGIN SOLUTION
        x = self.body(x)
        x = self.head(x)
        ### END SOLUTION
        return x


##--------------------------------------------------------
## model 34
##   Trunk:  
##    Head:  
##--------------------------------------------------------
class MyModel_v34(nn.Module):
    def __init__(self, dropout = True):
        super().__init__()
        self.dropout = dropout
        if self.dropout:
            self.name = "MyModel_v34"
        else:
            self.name = "MyModel_v34_No-DO"
        
        
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=3 , out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), 
            nn.ReLU(),
            # nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Dropout(p=0.2) if self.dropout else nn.Identity(), 
            
            # nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512), 
            # nn.ReLU(),
            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(p=0.2) if self.dropout else nn.Identity(), 
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(256 * 4* 4, 256 * 4* 4),
            nn.ReLU(),   
            nn.Linear(256 * 4 * 4, 2048 * 2),
            nn.ReLU(),   
            nn.Dropout(p=0.2) if self.dropout else nn.Identity(), 
            nn.Linear(2048 * 2, 10)
        )

    def forward(self, x):
        ### BEGIN SOLUTION
        x = self.body(x)
        x = self.head(x)
        ### END SOLUTION
        return x

##--------------------------------------------------------
## model 41
##   Trunk:  
##    Head:  
##--------------------------------------------------------
class Model_v41(nn.Module):
    
    def __init__(self, dropout=True):
        super().__init__()
        self.dropout = dropout
        if self.dropout:
            self.name = "Model_v41"
        else:
            self.name = "Model_v41_No-DO"
        l1_out = 64
        l2_out = 128
        l3_out = 256
        l4_out = 512
        l5_out = 1024
        l6_out = 2048
        p2 = 1
        p1 = 2
        
        # convolution layers
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels= l1_out, kernel_size=7, padding = p1),
            # nn.BatchNorm2d(l1_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=l1_out, out_channels=l2_out, kernel_size=5, padding = p2),
            nn.BatchNorm2d(l2_out),            
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=l2_out, out_channels=l3_out, kernel_size=5, padding = p2),
            nn.BatchNorm2d(l3_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=l3_out, out_channels=l4_out, kernel_size=5, padding = p2),
            nn.BatchNorm2d(l4_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=l4_out, out_channels=l5_out, kernel_size=5, padding = p2),
            nn.BatchNorm2d(l5_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),     

            nn.Conv2d(in_channels=l5_out, out_channels=l6_out, kernel_size=3, padding = p2),
            nn.BatchNorm2d(l6_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                 
        )

        
        # Fully connected layers
        self.head = nn.Sequential(
            nn.Linear(in_features= l6_out * 2 * 2, out_features=4096), 
            nn.ReLU(inplace=True),
            nn.Linear(in_features= 4096, out_features= 2048), 
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
## model 42
##   Trunk:  
##    Head:  
##--------------------------------------------------------
class Model_v42(nn.Module):
    
    def __init__(self, dropout=True):
        super().__init__()
        self.dropout = dropout
        if self.dropout:
            self.name = "Model_v42"
        else:
            self.name = "Model_v42_No-DO"
        l1_out = 64
        l2_out = 128
        l3_out = 256
        l4_out = 512
        l5_out = 1024
        l6_out = 1024
        p2 = 1
        p1 = 2
        
        # convolution layers
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels= l1_out, kernel_size=7, padding = p1),
            # nn.BatchNorm2d(l1_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=l1_out, out_channels=l2_out, kernel_size=5, padding = p2),
            nn.BatchNorm2d(l2_out),            
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=l2_out, out_channels=l3_out, kernel_size=5, padding = p2),
            nn.BatchNorm2d(l3_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=l3_out, out_channels=l4_out, kernel_size=5, padding = p2),
            nn.BatchNorm2d(l4_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=l4_out, out_channels=l5_out, kernel_size=5, padding = p2),
            nn.BatchNorm2d(l5_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),     

            nn.Conv2d(in_channels=l5_out, out_channels=l6_out, kernel_size=3, padding = p2),
            nn.BatchNorm2d(l6_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),           

            nn.Flatten()            
        )

        
        # Fully connected layers
        self.head = nn.Sequential(
            nn.Linear(in_features= l6_out * 2 * 2, out_features=4096), 
            nn.ReLU(inplace=True),
            nn.Linear(in_features= 4096, out_features= 2048), 
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
## model 43
##   Trunk:  
##    Head:  
##--------------------------------------------------------
class Model_v43(nn.Module):
    
    def __init__(self, dropout=True):
        super().__init__()
        self.dropout = dropout
        if self.dropout:
            self.name = "Model_v43"
        else:
            self.name = "Model_v43_No-DO"
        l1_out = 64
        l2_out = 256
        l3_out = 512
        l4_out = 1024
        l5_out = 2048
        l6_out = 2048
        p2 = 2
        p1 = 1
        k1 = 7
        k2 = 5 
        
        # convolution layers
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels= l1_out, kernel_size=7, padding = p2),
            nn.BatchNorm2d(l1_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=l1_out, out_channels=l2_out, kernel_size=5, padding = p1),
            nn.BatchNorm2d(l2_out),            
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=l2_out, out_channels=l3_out, kernel_size=5, padding = p1),
            nn.BatchNorm2d(l3_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=l3_out, out_channels=l4_out, kernel_size=5, padding = p1),
            nn.BatchNorm2d(l4_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=l4_out, out_channels=l5_out, kernel_size=5, padding = p1),
            nn.BatchNorm2d(l5_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),     

            # nn.Conv2d(in_channels=l5_out, out_channels=l6_out, kernel_size=3, padding = 0),
            # nn.BatchNorm2d(l6_out),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),          
            
            nn.Flatten()
        )

        
        # Fully connected layers
        self.head = nn.Sequential(
            nn.Linear(in_features= l5_out * 5* 5, out_features=4096), 
            nn.ReLU(inplace=True),
            nn.Linear(in_features= 4096, out_features= 2048), 
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
## model 44
##   Trunk:  
##    Head:  
##--------------------------------------------------------
class Model_v44(nn.Module):
    
    def __init__(self, dropout=True):
        super().__init__()
        self.dropout = dropout
        if self.dropout:
            self.name = "Model_v44"
        else:
            self.name = "Model_v44_No-DO"
        l1_out = 64
        l2_out = 256
        l3_out = 512
        l4_out = 1024
        # l5_out = 2048
        # l6_out = 2048
        p2 = 2
        p1 = 1
        k1 = 7
        k2 = 5 
        
        # convolution layers
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels= l1_out, kernel_size=7, padding = p2),
            nn.BatchNorm2d(l1_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),
            
            nn.Conv2d(in_channels=l1_out, out_channels=l2_out, kernel_size=5, padding = p1),
            nn.BatchNorm2d(l2_out),            
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),
            
            nn.Conv2d(in_channels=l2_out, out_channels=l3_out, kernel_size=5, padding = p1),
            nn.BatchNorm2d(l3_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),

            nn.Conv2d(in_channels=l3_out, out_channels=l4_out, kernel_size=5, padding = p1),
            nn.BatchNorm2d(l4_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            # nn.Conv2d(in_channels=l4_out, out_channels=l5_out, kernel_size=5, padding = p1),
            # nn.BatchNorm2d(l5_out),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),     

            # nn.Conv2d(in_channels=l5_out, out_channels=l6_out, kernel_size=3, padding = 0),
            # nn.BatchNorm2d(l6_out),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),          
            
            nn.Flatten()
        )
        
        # Fully connected layers
        self.head = nn.Sequential(
            nn.Linear(in_features= l4_out * 2* 2, out_features=2048), 
            nn.ReLU(inplace=True),
            nn.Linear(in_features= 2048, out_features= 1024), 
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=3)
        )
    
    def forward(self, x):
        
        # apply feature extractor
        x = self.body(x)
        # apply classification head
        x = self.head(x)      
        return x

        
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
