# Deep Reinforcement Learning In Action

Code Snippets from the [Deep Reinforcement Learning in Action](https://www.manning.com/books/deep-reinforcement-learning-in-action) book from Manning, Inc

## How this is Organized

The code snippets, listings, and projects are all embedded in Jupyter Notebooks
organized by chapter. Visit [http://jupyter.org/install](http://jupyter.org/install) for
instructions on installing Jupyter Notebooks.

We keep the original Jupyter Notebooks in their respective chapter folders. As we discover errata, we update notebooks in the Errata folder, so those notebooks are the most up-to-date in terms of errors corrected, but we keep the original Jupyter Notebooks to match the book code snippets.

## Requirements

In order to run many of the projects, you'll need at least the [NumPy](http://www.numpy.org/) library
and [PyTorch](http://pytorch.org/).

```
pip install -r requirements.txt
```

## Special Instructions
In the notebook 9, there's an issue (appearing in the 15th cell) you can solve by following the instructions of @scottmayberry in Farama-Foundation/MAgent2#14. That means to copy all the files and folders from https://github.com/Farama-Foundation/MAgent2/tree/main/magent2 to the local folder <venv_folder>/lib/python3.X/site-packages/magent2 (or similar path if your OS is other than Linux) - Thanks to [donlaiq](https://github.com/donlaiq) for this

## Contribute

If you experience any issues running the examples, please file an issue.
If you see typos or other errors in the book, please edit the [Errata.md](https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction/blob/master/Errata.md) file and create a pull request.
