# Errata
Here we keep an updated list of book errata listed by chapter. The errata will be addressed periodically in the eBook and liveBook versions and less frequently in print versions. We highly recommend using the GitHub code when following along the book as it is very difficult to maintain code within the book text.

## Chapter 2

### Section 2.4.1
Code should be:
>>> x = torch.Tensor([2,4]) #input data
>>> m = torch.randn(2, requires_grad=True) #parameter 1
>>> b = torch.randn(1, requires_grad=True) #parameter 2
>>> y = m*x+b #linear model
>>> y_known = torch.Tensor([5,9])
>>> loss = (torch.sum(y_known - y))**2 #loss function
>>> loss.backward() #calculate gradients
>>> m.grad
tensor([ -51.9402, -103.8803])

### Listing 2.10
Code should be:
def train(env, epochs=10000, learning_rate=1e-3):

### Section 2.5
Text should say: “When we train this network for 10,000 epochs”

### In Summary, in bullet point 5
“with probability ε – 1” should be “with probability 1 - ε “

## Chapter 3
### Listing 3.7
model2=model2= copy.deepcopy(model)
### Listing 3.8 
First line of code should be: from IPython.display import clear_output


## Chapter 4


## Chapter 8
Text should say: "In this chapter we will learn" instead of "In this chapter you will we learn"
Figure 8.14: Forward model should have an input for the actions and be concatenated with the encoded state.
### Listing 8.11
Code: The if statement related to the use of extrinsic reward should be fixed, "if use_extrinsic" not "if use_explicit", also in its relative balloon annotation. 

## Appendix
In text above the $f(x,y)=(Ax+By,Cx+Dy)$ should be corrected to "To find $\dot{x}$" from "To find $x$"







