First Version - Regression___________________________________________________________________________

In the first model I madee I predicted humidity based on temprature and discovered that an inc in temp will lead to a dec in humidity (which kinda seems wrong but ok)
I also wanna try using MAE instead of MSE to see how that changes it.

I will make another model which will guess either:
1. apparent temp from temp,humid,windspd,pressure
2. or temp from humid,windspd,pressure

idk if ill be able to make classification as well T_T


The first model, I started using the normal iterative method to do the prediction but I ended up doing a matrix method(yes with Claude but i understood the concept)

I noticed that the line got different slope and bias based on my number of iterations and step size on gradient descent
(
When I put n_iteration = 200000 and learning_rate at 0.001 I get a line that looks like it could apply to the data but if I make n_iteration = 20000 suddenly the line is exactly the opposite of the trend but if I increase step again then the line is good again
)

Second One - multilinear regression_____________________________________________________________________________________________
Im added a bunch of things in this one you can do either a MAE or MSE model, but it'll show u the difference between both anyway. It'll just train based on one or the other.
a bunch of stuff its hard to think off at 2 AM. rewrote the whole thing by hand (except the last few blocks of code tho) made everything more efficient and switched to panadas to read csv.
It shows loss curve, predicted vs actual and a residual plot that tells you if a linear combination even works for a relationship (I think)


_____________________________________________________________________________--
