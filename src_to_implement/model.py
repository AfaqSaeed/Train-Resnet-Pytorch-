import torch.nn as nn

class ResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualLayer, self).__init__()##
        # First convolutional layer
        self.convlayer1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False) ##
        self.bnlayer1 = nn.BatchNorm2d(out_channels) # Batch Norm Layer 1
        self.relu = nn.ReLU(inplace=True) # Just a normal relu activation 
        # Second convolutional layer
        self.convlayer2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False) ###
        self.bnlayer2 = nn.BatchNorm2d(out_channels) # Batch Norm Layer 2 
        self.skiplayer = nn.Sequential() # Just add an empty sequential Layer for realising the skip connection 
        # Skip connection
        if stride != 1 or in_channels != out_channels: ## handle an exception case where input channels anad output channels are not equal 
            self.skiplayer = nn.Sequential( 
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), # add a Conv2d and batchnorm to keep the output consitent 
                #and also no collisions in dimensions occur 
                nn.BatchNorm2d(out_channels) ### Some works 
            )

    def forward(self, input):
        # Forward pass through the residual block
        output= self.bnlayer2(self.convlayer2(self.relu(self.bnlayer1(self.convlayer1(input))))) ## Normal Forward Pass output 
        output+= self.skiplayer(input) # output with the skip connection for the resnet layers 
        output= self.relu(output) # pass it through the relu activation function
        return output
    
class ResNet(nn.Module):
    def __init__(self, reslayer=ResidualLayer, num_classes=2):
       
        super(ResNet, self).__init__()
        self.in_channels = 64
        # Initial convolutional layer
        self.convlayer1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bnlayer1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpoolinglayer = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Residual layers
        self.reslayer1 = self.make_layer(reslayer, 64)
        self.reslayer2 = self.make_layer(reslayer, 128, stride=2)
        self.reslayer3 = self.make_layer(reslayer, 256, stride=2)
        self.reslayer4 = self.make_layer(reslayer, 512, stride=2)
        self.avgpoolinglayer = nn.AdaptiveAvgPool2d((1, 1))
        self.fullyconnectedlayer = nn.Linear(512, 2)
        self.sigmoid = nn.Sigmoid()
    def make_layer(self, unit, out_channels, stride=1):
        """
        Create a layer of residual units.
        """
        layers = []
        layers.append(unit(self.in_channels, out_channels, stride))
        self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, input):
        """
        Forward pass of the ResNet model.
        """
        ## Normal convolution plus batchnorm for feature extraction
        output= self.maxpoolinglayer(self.relu(self.bnlayer1(self.convlayer1(input))))
        ### Pass through all residual layers 
        output= self.reslayer1(output)
        output= self.reslayer2(output)
        output= self.reslayer3(output)
        output= self.reslayer4(output)
        # apply average pooling to the output 
        output= self.avgpoolinglayer(output)
        ## flatten the output 
        output= output.view(output.size(0), -1)
        output= self.fullyconnectedlayer(output)
        output= self.sigmoid(output)
        return output