::::::::::::::::::::::::::::::::::::::::::::GENOME:::::::::::::::::::::::::::::::::::::::::::::::::::::

First I made the genome SVM. It has a linear and rbf kernel. I had made a default parameter baseline. then using cross variations over 5 fold (ahhhh) I found
the 'Optimal parameters'. None of the models overfitted which you can tell be looking at accuracy of test and train both. Accuracy/F1(HM)/ROC-AUC which all agree.
The RBF won overall.

checked for duplicate rows: 28.7% of the dataset is duplicated, so 51.7% of test rows also appear in training, 
which means my test scores are optimistic and the RBF-vs-linear gap is more reliable than either absolute number

the differnce between hyperparameters is negligible so the real difference is between each model.
 
On this dataset. RBF beat out linear. As you can see by the graphs as well as the data;
The linear kernel just doesn't have the capacity to fit the data set (which you figure out intuitvelty by looking at the graph (looks like a spiral));
RBF has the same values as linear when gamma = 0.01 ;
some of the data is same in both test and train so theres leakege;

I wanted to do a from scratch implementation but I didnt have the time. I would wanna come back to make this better but Midsems and other deadlines do not leave me 
much room for creativity. 

########################################pulasarrrrrrrrrrrr################################################


