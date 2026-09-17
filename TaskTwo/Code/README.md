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

well I speedran that but ok.
I did linear, poly, and rbf in this. Polynomial took like an hour to run once so no way am i doing that each time.
I added some graphs after claude said this could be some useful ones but i didnt really get them/though were too extra so i cut them out.


from what I understand from the data is that when I went through it all was that, when I used balanced weights the models missed less pulsars but got more wrong as well. which decreased the F1(Harmonic mean) but irl you'd rather get more wrong answers than miss actual correct ones.

also the polynomial kernel performed slightly worse than the other two
Performance depended overwhelmingly on coef0
rather than on C, γ or degree: at matched settings, switching coef0 from 1 to 0 cost up to 0.29 F1 (0.8115 → 0.5203 at C=0.1, γ=0.01, degree 3), and every polynomial configuration in the top thirty had coef0=1

Kernel choice swung F1 by 0.08 on gene expression and 0.005 on pulsar. Genome data set was curved but this is linearly sperable in 8D
As gaama tends to 0 the RBF kernel loses curvature and degenerates to linear
 









Confusion matrix with weights = balanced
 RBF (default)
[[2224   51]
 [  22  209]]
 LINEAR (default)
[[2212   63]
 [  25  206]]
 Confusion matrix without weights = balanced
RBF (default)
[[2265   10]
 [  41  190]]   
 LINEAR (default)
[[2268    7]
 [  45  186]]

ALSO if you wanna see it look like a actual table use raw mode or wtv its called

BELOW IS THE READINGS I GOT WITH THE POLYNOMIAL KERNEL BUT THEY'RE TOO HEAVY TO COMPUTE MULTIPLE TIMES SO I'LL REMOVE THEM FROM THE FINAL SUBMISSION.
________________________________________________________________
 LINEAR (default)
[[2268    7]
 [  45  186]]
              precision    recall  f1-score   support

           0       0.98      1.00      0.99      2275
           1       0.96      0.81      0.88       231

    accuracy                           0.98      2506
   macro avg       0.97      0.90      0.93      2506
weighted avg       0.98      0.98      0.98      2506

ROC-AUC : 0.966
PR-AUC  : 0.9147
train acc: 0.9758 | test acc: 0.9792
n_support: [315 309]

 RBF (default)
[[2265   10]
 [  41  190]]
              precision    recall  f1-score   support

           0       0.98      1.00      0.99      2275
           1       0.95      0.82      0.88       231

    accuracy                           0.98      2506
   macro avg       0.97      0.91      0.94      2506
weighted avg       0.98      0.98      0.98      2506

ROC-AUC : 0.9623
PR-AUC  : 0.9014
train acc: 0.9781 | test acc: 0.9796
n_support: [361 327]

 POLY (default)
[[2267    8]
 [  48  183]]
              precision    recall  f1-score   support

           0       0.98      1.00      0.99      2275
           1       0.96      0.79      0.87       231

    accuracy                           0.98      2506
   macro avg       0.97      0.89      0.93      2506
weighted avg       0.98      0.98      0.98      2506

ROC-AUC : 0.9609
PR-AUC  : 0.9161
train acc: 0.9754 | test acc: 0.9777
n_support: [321 305]
Fitting 5 folds for each of 69 candidates, totalling 345 fits

best params: {'svc__C': 10, 'svc__gamma': 'scale', 'svc__kernel': 'rbf'}
best CV f1 : 0.8763

 BEST MODEL ON TEST 
[[2265   10]
 [  40  191]]
              precision    recall  f1-score   support

           0       0.98      1.00      0.99      2275
           1       0.95      0.83      0.88       231

    accuracy                           0.98      2506
   macro avg       0.97      0.91      0.94      2506
weighted avg       0.98      0.98      0.98      2506

ROC-AUC : 0.9566
PR-AUC : 0.8999
n_support: [331 291]
   param_svc__kernel  param_svc__C param_svc__gamma  param_svc__degree  param_svc__coef0  mean_train_score  mean_test_score  rank_test_score
0                rbf         10.00            scale                NaN               NaN          0.893622         0.876289                1
1                rbf         10.00              0.1                NaN               NaN          0.889203         0.872517                2
2               poly          0.10                1                3.0               1.0          0.890625         0.872375                3
3               poly         10.00            scale                3.0               1.0          0.889169         0.872066                4
4                rbf        100.00              0.1                NaN               NaN          0.903279         0.872060                5
5               poly          1.00                1                3.0               1.0          0.899364         0.871508                6
6               poly         10.00                1                2.0               1.0          0.878193         0.869981                7
7               poly         10.00              0.1                2.0               1.0          0.876061         0.869616                8
8               poly         10.00            scale                2.0               1.0          0.876995         0.868513                9
9                rbf        100.00             0.01                NaN               NaN          0.876579         0.867656               10
10              poly          1.00                1                2.0               1.0          0.876945         0.867099               11
11              poly         10.00              0.1                3.0               1.0          0.887485         0.867005               12
12              poly         10.00                1                3.0               0.0          0.893325         0.865888               13
13               rbf          1.00                1                NaN               NaN          0.908658         0.864690               14
14              poly         10.00             0.01                3.0               1.0          0.867112         0.864492               15
15               rbf        100.00            scale                NaN               NaN          0.907971         0.864453               16
16              poly          1.00              0.1                2.0               1.0          0.866439         0.864290               17
17              poly          1.00            scale                3.0               1.0          0.880954         0.863787               18
18              poly          1.00                1                3.0               0.0          0.886758         0.863580               19
19              poly         10.00                1                3.0               1.0          0.903301         0.863387               20
20              poly          1.00              0.1                3.0               1.0          0.876999         0.863290               21
21              poly          1.00            scale                2.0               1.0          0.869302         0.863264               22
22              poly          0.10                1                2.0               1.0          0.871761         0.862773               23
23               rbf         10.00             0.01                NaN               NaN          0.864706         0.861463               24
24               rbf          1.00            scale                NaN               NaN          0.871148         0.860265               25
25               rbf          1.00              0.1                NaN               NaN          0.868054         0.858885               26
26              poly          0.10                1                3.0               0.0          0.877286         0.858321               27
27              poly         10.00            scale                3.0               0.0          0.868307         0.856655               28
28              poly         10.00             0.01                2.0               1.0          0.860280         0.855945               29
29              poly         10.00              0.1                3.0               0.0          0.863262         0.852972               30
30            linear        100.00              NaN                NaN               NaN          0.855993         0.852684               31
31            linear         10.00              NaN                NaN               NaN          0.855656         0.852684               31
32            linear          1.00              NaN                NaN               NaN          0.854681         0.851367               33
33               rbf         10.00                1                NaN               NaN          0.940594         0.851241               34
34            linear          0.10              NaN                NaN               NaN          0.848181         0.845871               35
35              poly          0.10            scale                3.0               1.0          0.855112         0.844809               36
36              poly          0.10              0.1                3.0               1.0          0.850944         0.841675               37
37              poly         10.00                1                2.0               0.0          0.852327         0.841028               38
38              poly          1.00             0.01                3.0               1.0          0.841104         0.839881               39
39               rbf          1.00             0.01                NaN               NaN          0.839367         0.839485               40
40               rbf          0.10            scale                NaN               NaN          0.841555         0.838518               41
41              poly          1.00                1                2.0               0.0          0.849555         0.838377               42
42              poly          0.10            scale                2.0               1.0          0.841934         0.838252               43
43              poly          0.10              0.1                2.0               1.0          0.839707         0.837533               44
44               rbf          0.10              0.1                NaN               NaN          0.839039         0.836445               45
45              poly         10.00            scale                2.0               0.0          0.843025         0.835015               46
46              poly          1.00            scale                3.0               0.0          0.851206         0.832522               47
47              poly          1.00             0.01                2.0               1.0          0.836203         0.831670               48
48               rbf        100.00                1                NaN               NaN          0.965980         0.831468               49
49              poly          0.10                1                2.0               0.0          0.841130         0.829983               50
50              poly         10.00              0.1                2.0               0.0          0.841130         0.829983               50
51              poly          1.00              0.1                3.0               0.0          0.842634         0.828529               52
52            linear          0.01              NaN                NaN               NaN          0.827418         0.823840               53
53              poly          0.10            scale                3.0               0.0          0.828127         0.819678               54
54              poly          0.10              0.1                3.0               0.0          0.818710         0.816032               55
55              poly          1.00            scale                2.0               0.0          0.820096         0.815618               56
56              poly          0.10             0.01                3.0               1.0          0.812693         0.811497               57
57              poly          1.00              0.1                2.0               0.0          0.815058         0.810781               58
58               rbf          0.10             0.01                NaN               NaN          0.808888         0.809375               59
59              poly          0.10             0.01                2.0               1.0          0.808325         0.806354               60
60              poly          0.10            scale                2.0               0.0          0.799976         0.796442               61
61              poly         10.00             0.01                2.0               0.0          0.797215         0.794323               62
62              poly          0.10              0.1                2.0               0.0          0.797215         0.794323               62
63              poly         10.00             0.01                3.0               0.0          0.777703         0.773536               64
64              poly          1.00             0.01                2.0               0.0          0.741715         0.737616               65
65              poly          1.00             0.01                3.0               0.0          0.672229         0.668090               66
66               rbf          0.10                1                NaN               NaN          0.693676         0.635116               67
67              poly          0.10             0.01                2.0               0.0          0.619190         0.616380               68
68              poly          0.10             0.01                3.0               0.0          0.524392         0.520302               69

best per kernel:
   param_svc__kernel  param_svc__C param_svc__gamma  param_svc__degree  param_svc__coef0  mean_train_score  mean_test_score  rank_test_score
30            linear         100.0              NaN                NaN               NaN          0.855993         0.852684               31
2               poly           0.1                1                3.0               1.0          0.890625         0.872375                3
0                rbf          10.0            scale                NaN               NaN          0.893622         0.876289                1


