# Ongoing changes and future plans

In this page we will attempt to document the main ongoing changes in terms of model development, and the main features we are planning for the future.

- We are currently working on ensuring full end-to-end differentiability of the whole ODINN ecosystem with `Enzyme.jl`. We are very close to achieving this. A major release, including other ongoing features will be announced once everything is properly integrated and tested.

- We have a new interface to declare laws (i.e. empirical laws/parametrizations) to parametrize subparts of an ice flow model. These consist on a combination of input types, representing different input variables for a given law, and a regressor, which will parametrize a certain variable given those inputs. We are finishing the design and testing of this interface, which should be shortly announced and properly documented. 

- We have plans to host all the preprocessed glacier directories in a server, so users can automatically download them without having to preprocess them using `Gungnir`. 