################################################### XOR MLP##############################################
SO YEAHHHH IMADE THE XOR MLP
The math I feel like is pretty easy but trning it to python was the annoying part. I took a lot of help, but I understood all the code. I played around with
the random seed and thought the differnt plots looked cool so I added the option of simply putting them in by input.
ALso yeah it was 2-4-1 where 2 was input and the 4,1 did actual computations. 1 is the output layer ofc.
Idek what I should mention here the math feels pretty easy to me for this. I used the Cross Entropy as its easy to compute with Sigmoid. I wanted to try the ReLU too
cos the doctor at the end of 3B1B said its the preffered one now cos of reasons but ill do that on my own later now(Also Tanh) Theres also stuff like softmax . 
Also have to finish PyTorch.
I didnt do anything fancy most of what and why is mentioned as coments in the code itself.
I guess I did the initialisation where I divided by root2 and root 4 based on matrix size, so that the values of weights are not saturating the sigmoid at the start
Loss kinda sits around for a while when its at ln2. 


#############################################MNiST Hell#####################################################
So the MNIST Neural network, I did (input)784-128-64-10(output) 
I mainly did another layer cos I wanted to try multiple layers. 
I did ReLU as the activation functions for the hidden layers, cross-entropy for the error function, softmax got applied internally. Did SGD to cut computation.
weird ass idx files btw eh
During loss for validation was lowest at epoch 4 but trainin kept falling which would show it started memorizing instead of generalizing
I played around with the number of epochs and learing rate mainly lowering lr and increasing epoch. I also played with the seed,
and got around the same accuracy mostly below so I think the max capablity the model has is 97.94 only.
Also I took 16 as the seed cos my bday
ALSO, it was 97.94% accurate so yeah

almost all the mistake ones included that damned 0 that looks like a 6. Its not even the models fault, altho it made some pretty bad mistakes a few times.
THe model knows no clue of which pixels are neighbors which are years apart so it cant tell apart why wed think theyre obv diff. I think.
I kinda got lost in playing with the epochs

Honestly I did get hekp from CLaude, PyTorch is very confusing to me. But I went block by block asked it to explain and did my best to understand whats what.
