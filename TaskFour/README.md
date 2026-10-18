I Made a 3072-1024-512-128-10 MLP.
unfortunately Ive been sick since yesterday so I was late and I didnt complete a from-scratch version. I'll try to submit by tmr-dayfter

This MLP I submited slowly evolved from first having just SGD to different optimiers and hyperparameters by user input.
then I made two versions with just L2 and Dropout each and mad one with them together
FIanlly added multiple seeds option and just had claude make the code for the ploting of multiple seeds.

Used SGD/Momentum/AdamW

Uh the model started to underfit, it takes so much time to run the code and I deleted the previous runs so I dont have actuall numbers except one run, but I ran 15 epochs AdamW 0.01 wd and 0.3 dropout and val acc was 45.48% & val loss 1.5455, train loss 1.5962 & train acc 42.69% and TEST | loss 1.5252 | accuracy 46.63% (4663/10000). It(accuracy) should be around mid 50s.

I'll soon add a run of 3 seeds with like 30 epochs and 0.5 dropout etc and show the numbers cos. Also add the scracth version cos ofc
well it took like an hour for that one run which I uploaded js now with 3 seeds. It was so bad, it didnt learn,the lr mightve been too much I though 30 epochs 0.002 wpuld be better apparently not, ot maybe it was dropout 0.4%
