
from torchsummary import summary
from Models import *

model = SYMNet(2, 2, 8)
summary(model, [(1, 160, 192),(1, 160, 192)])
# summary(model, [(1, 160, 192, 224),(1, 160, 192, 224)])
# summary(model, (1, 128,128))#,(1, 80, 96, 112)])