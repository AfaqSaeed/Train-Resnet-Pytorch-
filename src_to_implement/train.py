import torch as t
from data import ChallengeDataset
from torch.utils.data import DataLoader
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split
from model import ResNet,ResidualLayer

# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
# TODO
datafilepath = "./data.csv"
# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
# TODO
dataframe = pd.read_csv(datafilepath, sep=";")
traindf, testdf = train_test_split(dataframe, test_size=0.3)
traindata=ChallengeDataset(traindf,mode="train")
testdata=ChallengeDataset(testdf,mode="test")

train_dataloader = DataLoader(traindata, batch_size=32, shuffle=True)
test_dataloader = DataLoader(testdata,batch_size=8)

# test train dataloader
''' 
for (image,label) in train_dataloader:
    for index in range(2):
        plt.title(str(label[index,0])+","+ str(label[index,1]))
        plt.imshow(image[index].permute(1, 2, 0))
        plt.show()
'''
#test test dataloader 
''' 
for (image,label) in test_dataloader:
   #for index in range(2):
    plt.title(str(label[0])+","+ str(label[1]))
    plt.imshow(image.permute(1, 2, 0))
    plt.show()
'''
    

# create an instance of our ResNet model
model = ResNet(ResidualLayer,2)
"""
for (image_batch,label) in train_dataloader:
    
        output=model.forward(image_batch)
        print(output[:,0],label[:,0])
"""
# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
creiterion = t.nn.BCELoss()
# set up the optimizer (see t.optim)
optimizer = t.optim.Adam(params= model.parameters(),lr=0.001)
# create an object of type Trainer and set its early stopping criterion
cuda = t.cuda.is_available() 
print(t.version.cuda)
print("Cuda Available ",cuda)
trainer =Trainer(model,creiterion,optimizer,train_dataloader,test_dataloader,cuda,0.5)
# TODO

# go, go, go... call fit on trainer

res = trainer.fit(50)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')