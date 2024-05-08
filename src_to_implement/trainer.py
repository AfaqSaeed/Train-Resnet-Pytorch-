import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
import numpy as np 
import os


class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
            
    def save_checkpoint(self, epoch):
        if not (os.path.exists("./checkpoints")):
            os.mkdir("./checkpoints")
        t.save({'state_dict': self._model.state_dict()}, './checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    def save_checkpoint_f1 (self,epoch):
        if not (os.path.exists("./checkpoints_f1")):
            os.mkdir("./checkpoints_f1")
        t.save({'state_dict': self._model.state_dict()}, './checkpoints_f1/checkpoint_{:03d}.ckp'.format(epoch))
    def restore_checkpoint(self, epoch_n):
        ckp = t.load('./checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called. This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        self._optim.zero_grad()

        # -propagate through the network
        prediction = self._model(x)
        # -calculate the loss
        # print("Prediction",prediction.dtype,"Y Type",y.dtype)
        loss = self._crit(prediction,y)
        # -compute gradient by backward propagation
        loss.backward()
        # -update weights
        self._optim.step()
        
        # -return the loss
        return loss
        #TODO
        
        
    
    def val_test_step(self, x, y):
        
        
        # predict
        with t.no_grad():
            prediction = self._model(x)

        # propagate through the network and calculate the loss and predictions
        loss = self._crit(prediction,y)
        
        # return the loss and the predictions
        return loss , prediction
        #TODO
        
    def train_epoch(self):
        # set training mode
        self._model.train()
        sum_loss = 0
        # iterate through the training set
        for (image_batch,label) in tqdm(self._train_dl):
            if self._cuda:
                image_batch = image_batch.cuda()
                label = label.cuda()
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
            loss =self.train_step(image_batch,label)
            # perform a training step
            
            sum_loss = sum_loss + loss

        avg_loss = sum_loss/len(self._train_dl)
        return avg_loss
        # calculate the average loss for the epoch and return it
        #TODO
    
    def val_test(self):
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        self._model.eval()
  # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore. 
        # iterate through the validation set
        sum_loss = 0
        f1_crack_sum = 0
        f1_inactive_sum = 0
        crack_prediction_list =[]
        inactive_prediction_list=[]
        crack_label_list=[]
        inactive_label_list=[]
        for (image_batch,label) in tqdm(self._val_test_dl):
        # transfer the batch to the gpu if given
        
            if self._cuda:
                image_batch = image_batch.cuda()
                label = label.cuda()
        # perform a validation step
                
            loss,predictions =self.val_test_step(image_batch,label)
            sum_loss = sum_loss + loss
            threshold_pred = np.where(predictions.cpu().detach().numpy()>=0.5,1.0,0.0)
            t.cuda.empty_cache()

            # print("Threshold ",threshold_pred[:,0],"Label",label[:,0].cpu().detach().numpy())
            crack_prediction_list.extend(threshold_pred[:,0])
            inactive_prediction_list.extend(threshold_pred[:,1])
            crack_label_list.extend(label[:,0].cpu().detach().numpy())
            inactive_label_list.extend(label[:,1].cpu().detach().numpy())

            
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        avg_loss = sum_loss/len(self._val_test_dl)

        print(len(crack_prediction_list),len(crack_label_list),len(inactive_prediction_list),len(inactive_label_list))
        f1_crack_score = f1_score(crack_prediction_list,crack_label_list)
        f1_inactive_score = f1_score(inactive_prediction_list,inactive_label_list)
        f1_mean = (f1_crack_score + f1_inactive_score)/2
        # f1_crack_score = f1_crack/len(self._val_test_dl)
        # f1_inactive_score = f1_crack/len(self._val_test_dl)

        print("F1 Cracked :",f1_crack_score)
        print("F1 Inactive :",f1_inactive_score)
        print("F1 Mean",f1_mean)

        

        
        # return the loss and print the calculated metrics
        #TODO
        return avg_loss,f1_mean
        
    
    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch 
        train_losses  = []
        validation_losses = []
        #TODO
        epoch_no = 0
        best_loss = 10000000
        best_f1_Score = 0
        not_improve_counter = 0
        while True:
            
            # stop by epoch number
            if epoch_no > epochs:
                print(f"Stopped after {epochs} iterations")
                break
            # train for a epoch and then calculate the loss and metrics on the validation set
            train_loss =self.train_epoch()
            
            val_loss,f1_mean = self.val_test()
        
            print(f"Epoch_no : {epoch_no} Train_loss : {train_loss} ,Val_loss : {val_loss}")
        
            if val_loss < best_loss:
                best_loss = val_loss
                 # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
                self.save_checkpoint(epoch_no) 
            if best_f1_Score<f1_mean:
                best_f1_Score = f1_mean
                self.save_checkpoint_f1(epoch_no)
            # else:    
            #     if self._early_stopping_patience > (val_loss - best_loss):
            #         not_improve_counter = not_improve_counter + 1     
            #     if not_improve_counter>3:
            #         break


           # check whether early stopping should be performed using the early stopping criterion and stop if so
            
            # append the losses to the respective lists
            train_losses.append(train_loss)
            validation_losses.append(val_loss)
            epoch_no = epoch_no + 1
        return train_losses,validation_losses
            # return the losses for both training and validation
        #TODO
                    
        
        
        
