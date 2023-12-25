from typing import List
import torch
from torch import nn
import torchmetrics
import torch.nn.functional as F
import pytorch_lightning as pl
import transformers
from torch.optim.lr_scheduler import StepLR, LambdaLR, SequentialLR
from sentence_transformers import SentenceTransformer


# Basic model class with classification head
class Model(pl.LightningModule):
    def __init__(self, model_name, lr, loss_fns: List[torch.nn.Module]):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr

        # plm: pretrained language model
        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1
        )
        
        # special token의 embedding을 학습에 포함시킵니다.
        self.plm.resize_token_embeddings(self.plm.get_input_embeddings().num_embeddings + 5) # 야매로 5개 더 추가해줍니다.

        # Loss 계산을 위해 사용될 L1Loss를 호출합니다.
        self.loss_fns = loss_fns
        

    def forward(self, **x):
        x = self.plm(**x)['logits']

        return x
    

    def custom_loss(self, outputs, targets):
        if len(self.loss_fns) == 1:
            loss = self.loss_fns[0](outputs, targets)
        elif len(self.loss_fns) < 1:
            raise ValueError("At least one loss function should be defined.")
        else:
            loss = 0
            for loss_fn in self.loss_fns:
                loss += loss_fn(outputs, targets)
            loss /= len(self.loss_fns)
        return loss
    

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self(**x)
        loss = self.custom_loss(logits, y.float())
        self.log("train_loss", loss)

        return loss
    

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(**x)
        loss = self.custom_loss(logits, y.float())
        self.log("val_loss", loss)

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss
    

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(**x)

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

    
    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(**x)

        return logits.squeeze()

    # training_step 이전에 호출되는 함수입니다.
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        # Define the warm-up phase
        warmup_steps = 3
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: float(epoch) / warmup_steps if epoch < warmup_steps else 1)

        # Define the StepLR scheduler
        step_size = 2  # Number of epochs between each step
        gamma = 0.9     # Multiplicative factor of learning rate decay
        step_scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

        # Combine schedulers with SequentialLR
        schedulers = [warmup_scheduler, step_scheduler]
        milestones = [warmup_steps]  # The epochs at which to switch schedulers, here after warmup
        combined_scheduler = SequentialLR(optimizer, schedulers, milestones)

        return {"optimizer": optimizer, "lr_scheduler": combined_scheduler}
    

# Basic regression model class
class RegressionModel(pl.LightningModule):
    def __init__(self, model_name, lr, loss_fns: List[torch.nn.Module]):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr

        self.plm = transformers.AutoModel.from_pretrained(
            pretrained_model_name_or_path=model_name
        )

        self.regression_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.plm.config.hidden_size, 1),
        )
        
        self.plm.resize_token_embeddings(self.plm.get_input_embeddings().num_embeddings + 5)

        self.loss_fns = loss_fns

    def forward(self, **x):
        x = self.plm(**x)
        x = x.last_hidden_state[:, 0, :]
        x = self.regression_head(x)
        return x

    def custom_loss(self, outputs, targets):
        if len(self.loss_fns) == 1:
            loss = self.loss_fns[0](outputs, targets)
        elif len(self.loss_fns) < 1:
            raise ValueError("At least one loss function should be defined.")
        else:
            loss = 0
            for loss_fn in self.loss_fns:
                loss += loss_fn(outputs, targets)
            loss /= len(self.loss_fns)
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self(**x)
        loss = self.custom_loss(logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(**x)
        loss = self.custom_loss(logits, y.float())
        self.log("val_loss", loss)

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(**x)

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(**x)

        return logits.squeeze()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)

        warmup_steps = 3
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: float(epoch) / warmup_steps if epoch < warmup_steps else 1)

        step_size = 2
        gamma = 0.9
        step_scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

        schedulers = [warmup_scheduler, step_scheduler]
        milestones = [warmup_steps]
        combined_scheduler = SequentialLR(optimizer, schedulers, milestones)

        return {"optimizer": optimizer, "lr_scheduler": combined_scheduler}
    

# Regression model with utilizing hidden states of special tokens
class SpecialTokenRegressionModel(pl.LightningModule):
    def __init__(self, model_name, lr, loss_fns: List[torch.nn.Module]):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr

        self.plm = transformers.AutoModel.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1
        )

        self.regression_head = nn.Sequential(
            nn.Linear(3*self.plm.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(), # .5 default
            nn.Linear(128, 1),
        )
        
        self.plm.resize_token_embeddings(self.plm.get_input_embeddings().num_embeddings + 5) # 야매로 5개 더 추가해줍니다.

        self.loss_fns = loss_fns

    def forward(self, **x):
        x = self.plm(**x)
        x = x.last_hidden_state[:, 0:3, :] # [batch_size, 3, hidden_size] = [batch_size, 3, 768]
        x = x.view(x.size(0), -1) # [batch_size, 3*hidden_size] = [batch_size, 3*768]
        x = self.regression_head(x)
        return x

    def custom_loss(self, outputs, targets):
        if len(self.loss_fns) == 1:
            loss = self.loss_fns[0](outputs, targets)
        elif len(self.loss_fns) < 1:
            raise ValueError("At least one loss function should be defined.")
        else:
            loss = 0
            for loss_fn in self.loss_fns:
                loss += loss_fn(outputs, targets)
            loss /= len(self.loss_fns)
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self(**x)
        loss = self.custom_loss(logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(**x)
        loss = self.custom_loss(logits, y.float())
        self.log("val_loss", loss)

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(**x)

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))
    
    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(**x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        warmup_steps = 3
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: float(epoch) / warmup_steps if epoch < warmup_steps else 1)

        step_size = 2
        gamma = 0.9 
        step_scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

        schedulers = [warmup_scheduler, step_scheduler]
        milestones = [warmup_steps]
        combined_scheduler = SequentialLR(optimizer, schedulers, milestones)

        return {"optimizer": optimizer, "lr_scheduler": combined_scheduler}
    

# Regression model with RDrop panelty
class RDropRegressionModel(pl.LightningModule):
    def __init__(self, model_name, lr, loss_fns: List[torch.nn.Module]):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr
        self.rdrop_alpha = 0.2

        self.plm = transformers.AutoModel.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1
        )

        self.regression_head = nn.Sequential(
            nn.Linear(3*self.plm.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(), # .5 default
            nn.Linear(128, 1),
        )
        
        self.plm.resize_token_embeddings(self.plm.get_input_embeddings().num_embeddings + 5)

        self.loss_fns = loss_fns

    def forward(self, **x):
        x = self.plm(**x)
        x = x.last_hidden_state[:, 0:3, :] 
        x = x.view(x.size(0), -1)
        x = self.regression_head(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits1 = self(**x)
        logits2 = self(**x)
        loss = self.custom_loss(logits1, y.float())

        rdrop_reg = self.rdrop_loss(logits1, logits2, alpha=self.rdrop_alpha, method="l1")

        total_loss = loss + rdrop_reg

        self.log("train_loss", total_loss)
        return total_loss

    def custom_loss(self, outputs, targets):
        if len(self.loss_fns) == 1:
            loss = self.loss_fns[0](outputs, targets)
        elif len(self.loss_fns) < 1:
            raise ValueError("At least one loss function should be defined.")
        else:
            loss = 0
            for loss_fn in self.loss_fns:
                loss += loss_fn(outputs, targets)
            loss /= len(self.loss_fns)
        return loss

    def rdrop_loss(self, logits1, logits2, alpha=1.0, method="mse"):
        """
        Compute R-Drop regularization loss.
        
        Args:
            logits1 (torch.Tensor): Logits from the first forward pass.
            logits2 (torch.Tensor): Logits from the second forward pass.
            alpha (float): Weight of the R-Drop regularization term.
            method (str): The method for R-Drop loss ('mse' or 'l1').

        Returns:
            torch.Tensor: The R-Drop regularization loss.
        """
        if method == "mse":
            rdrop_reg = F.mse_loss(logits1, logits2)
        elif method == "l1":
            rdrop_reg = F.l1_loss(logits1, logits2)
        else:
            raise ValueError("Invalid method for R-drop loss. Use 'mse' or 'l1'.")
        
        return alpha * rdrop_reg
    

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(**x)
        loss = self.custom_loss(logits, y.float())
        self.log("val_loss", loss)

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(**x)

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))
    
    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(**x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)

        warmup_steps = 3
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: float(epoch) / warmup_steps if epoch < warmup_steps else 1)

        step_size = 2 
        gamma = 0.9    
        step_scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

        schedulers = [warmup_scheduler, step_scheduler]
        milestones = [warmup_steps]
        combined_scheduler = SequentialLR(optimizer, schedulers, milestones)

        return {"optimizer": optimizer, "lr_scheduler": combined_scheduler}
    

# Advanced model with cosine similarity
class SimilarityModel(pl.LightningModule):
    def __init__(self, model_name, lr, loss_fns: List[torch.nn.Module]):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr
        self.lamb = 0.5 # define ratio between electra model and sroberta model

        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name, num_labels=1)

        self.plm.resize_token_embeddings(self.plm.get_input_embeddings().num_embeddings + 5) # increase vocab size by 5 (the number of special tokens)

        self.sroberta = SentenceTransformer('jhgan/ko-sroberta-multitask')

        self.linear_1 = torch.nn.Linear(768, 768)
        self.linear_2 = torch.nn.Linear(768, 768)

        self.electra_loss_func = torch.nn.L1Loss()
        self.selectra_loss_func = torch.nn.MSELoss()
        self.rdrop_alpha = 0.1

        self.loss_fns = loss_fns

    def forward(self, **x):

        # x['input_ids_1'].size() = [batch_size, 1, max_length]
        # x['input_ids_2'].size() = [batch_size, 2, max_length]

        ### version 1 : simple forward pass ###
        electra_output = self.plm(x['input_ids_1'],x['attention_mask_1'])['logits'] # [batch_size, 1]

        ### version 2 : forward each sentence vectors to different linear layers and calculate cosine similarity ###
        with torch.inference_mode():
            for i in range(x['input_ids_2'].size()[0]): # iterate over batch_size (sentence by sentence)
                # x['input_ids_2'][i].size() : [2, max_length]   ==>   sroberta_output.size() : [2, 768]
                sent1_sent2_embeddings = self.sroberta({'input_ids':x['input_ids_2'][i],'attention_mask': x['attention_mask_2'][i]})['sentence_embedding'] # [2, 768]
                if i==0:
                    sroberta_output = sent1_sent2_embeddings.unsqueeze(0) # sroberta_output.size() : [1, 2, 768]
                else:
                    sroberta_output = torch.cat([sroberta_output,sent1_sent2_embeddings.unsqueeze(0)],0) # sroberta_output.size() : [i+1, 2, 768]
                
        # sroberta_output.size() : [batch_size, 2, 768]
        sroberta_output_sent1 = self.linear_1(sroberta_output[:,0,:]) # sroberta_output_sent1.size() : [batch_size, 768]
        sroberta_output_sent2 = self.linear_2(sroberta_output[:,1,:]) # sroberta_output_sent2.size() : [batch_size, 768]

        # sent1 & sent2 cosine similarity
        sroberta_outputs = sroberta_output_sent1.matmul(sroberta_output_sent2.transpose(0,1)).diagonal() # sroberta_output.size() : [batch_size]
        sroberta_output_norms =  torch.norm(sroberta_output_sent1,dim=1) * torch.norm(sroberta_output_sent2,dim=1) # sroberta_output_norms.size() : [batch_size]
            
        sroberta_final_outputs = sroberta_outputs / sroberta_output_norms # sroberta_final_outputs.size() : [batch_size]
        sroberta_final_outputs = sroberta_final_outputs.unsqueeze(1) # sroberta_final_outputs.size() : [batch_size, 1]

        return electra_output, sroberta_final_outputs # [batch_size, 1], [batch_size, 1]
    
    def rdrop_L1(self, logits_1, logits_2, alpha):
        return torch.abs(logits_1 - logits_2).mean() * alpha

    def training_step(self, batch, batch_idx):
        x, y = batch

        electra_outputs, sroberta_outputs = self(**x)

        electra_loss = self.electra_loss_func(electra_outputs, y.float())
        sroberta_loss = self.selectra_loss_func(sroberta_outputs, y.float())

        total_loss = (self.lamb)*electra_loss + (1-self.lamb)*sroberta_loss # ratio
        total_loss += self.rdrop_L1(electra_outputs, sroberta_outputs, self.rdrop_alpha)

        self.log("train_loss", total_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        electra_outputs, sroberta_outputs = self(**x)

        electra_loss = self.electra_loss_func(electra_outputs, y.float())
        sroberta_loss = self.selectra_loss_func(sroberta_outputs, y.float())

        logits = (self.lamb)*electra_outputs + (1-self.lamb)*sroberta_outputs

        total_loss = (self.lamb)*electra_loss + (1-self.lamb)*sroberta_loss
        total_loss += self.rdrop_L1(electra_outputs, sroberta_outputs, self.rdrop_alpha)

        self.log("val_loss", total_loss)

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return total_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        
        electra_outputs, sroberta_outputs = self(**x)
        logits = (self.lamb)*electra_outputs + (1-self.lamb)*sroberta_outputs

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x = batch
        
        electra_outputs, sroberta_outputs = self(**x)
        logits = (self.lamb)*electra_outputs + (1-self.lamb)*sroberta_outputs

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
