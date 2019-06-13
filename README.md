# Focal-loss
The code is tensorflow implement for focal loss for Dense Object Detection. https://arxiv.org/abs/1708.02002

# usage
logits->[B,H, W, 1] # NOT sigmoid operation  
labels->[B,H, W]  
weights->[B,H, W] # can be None  
Focal= FocalLoss(alpha=0.25, gamma=2)  
loss = Focal.get_loss(logits, labels, weights=None)  

 
