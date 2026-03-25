import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchinfo import summary

# model 0.0 

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
        
# model 1.2

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


# model 1.3

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

# model 1.4

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

# model 1.5

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

# model 1.6

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

# model 1.7

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

                
## model 1.8

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

                
## model 1.9

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

                
## model 2.0

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

                
## model 2.1

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


                
## model 2.2

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
## model 40
##--------------------------------------------------------
class MyModel_v40(nn.Module):
    def __init__(self, dropout = True):
        super().__init__()
        self.dropout = dropout
        if self.dropout:
            self.name = "MyModel_v40"
        else:
            self.name = "MyModel_v40_No-DO"
        pad = 0
        dil = 2
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
            
            nn.Dropout(p=0.2) if self.dropout else nn.Identity(), 
            
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding = padding),
            nn.BatchNorm2d(512), 
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding = padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding = padding),
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

            nn.Linear(2048 * 2, 3)
        )

    def forward(self, x):
        ### BEGIN SOLUTION
        x = self.body(x)
        x = self.head(x)
        ### END SOLUTION
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