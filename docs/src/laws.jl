# # Laws tutorial

# This tutorial provides simple examples on how to create learnable and non learnable 
# laws and how to inject them into the iceflow model.

# Let's say we have followed the classical workflow from ODINN, shown in the Forward simulation and Functional inversion tutorials. When we declare the `Model` type, we can specify the laws that we want to use in the iceflow model. Here we will briefly show how to do it. For more details you can check the [Understanding the Law interface section](./inversions.md).

# ## Learnable laws

# Learnable laws are laws that can be trained using a regressor. They are used to map input variables to a target variable in the iceflow model. 
# In ODINN, we have implemented several learnable laws that can be used in the iceflow model.

using ODINN

## Dummy parameters, only specifying the type of loss function to be used
params = Parameters(UDE = UDEparameters(empirical_loss_function=LossH())) 
nn_model = NeuralNetwork(params)

A_law = LawA(nn_model, params)

model = Model(
    iceflow = SIA2Dmodel(params; A=A_law),
    mass_balance = TImodel1(params; DDF=6.0/1000.0, acc_factor=1.2/1000.0),
    regressors = (; A=nn_model)
)

# The `LawA` law is a learnable law that maps the long term air temperature `T` to the creep coefficient `A`.
# It is defined as a neural network that takes as input the long term air temperature `T` and returns the creep coefficient `A`.
# The parameters θ of the neural network are learned during the inversion process, by minimizing the loss function given some target data (for this case the ice thickness).

# ## Non learnable laws

# Non learnable laws are laws that are not trained using a regressor. They are used to map input variables to a target variable in the iceflow model, but they do not have any learnable parameters.

# Here is a quick example also drawn from the Functional Inversions tutorial. We define a synthetic law to generate the synthetic dataset. For this, we use some tabular data from Cuffey and Paterson (2010).

using ODINN

params = Parameters() # dummy parameters

A_law = CuffeyPaterson()

model = Model(
    iceflow = SIA2Dmodel(params; A=A_law),
    mass_balance = TImodel1(params; DDF=6.0/1000.0, acc_factor=1.2/1000.0),
)

# In this ice flow model, the ice rigidity `A` is defined by the `CuffeyPaterson` law, which is a non-learnable law that maps the long term air temperature `T` to the creep coefficient `A`. 