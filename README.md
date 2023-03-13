# ML
Developer writes the function in regular programming which takes input and produces output.
Machine Learning is computer writing the function (model/algorithm) based on input and output.

https://github.com/mrdbourke/zero-to-mastery-ml

https://discord.com/
https://zerotomastery.io/
https://www.youtube.com/@ZeroToMastery

https://teachablemachine.withgoogle.com/
https://ml-playground.com/#
https://www.elementsofai.com/

https://www.mrdbourke.com/a-6-step-field-guide-for-building-machine-learning-projects/

https://www.mrdbourke.com/get-your-computer-ready-for-machine-learning-using-anaconda-miniconda-and-conda/
https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html
https://docs.conda.io/en/latest/

https://docs.conda.io/en/latest/miniconda.html

Open Anaconda Prompt
```
cd ~\ml\sample_project
conda create --prefix ./env pandas numpy matplotlib scikit-learn
```

To activate this environment, use `conda activate ~\ml\sample_project\env`
To deactivate an active environment, use `conda deactivate`

```
conda activate ~\ml\sample_project\env
conda install jupyter
conda install seaborn
jupyter notebook
```

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
```

```
conda deactivate
conda env list
conda list
conda search scikit-learn --info
conda update scikit-learn
```

export current environment: `conda env export > environment.yml`
create new environment using exported yaml: `conda env create --prefix ./env -f ./environment.yml`

Sharing your Conda Environment
There may come a time where you want to share the contents of your Conda environment.
This could be to share a project workflow with a colleague or with someone else who's trying to set up their system to have access to the same tools as yours.
There a couple of ways to do this:
1. Share your entire project folder (including the environment folder containing all of your Conda packages).
2. Share a .yml (pronounced YAM-L) file of your Conda environment.

The benefit of 1 is it's a very simple setup, share the folder, activate the environment, run the code. However, an environment folder can be quite a large file to share.

That's where 2 comes in. A .yml is basically a text file with instructions to tell Conda how to set up an environment.

For example, to export the environment we created earlier at /Users/daniel/Desktop/project_1/env as a YAML file called environment.yml we can use the command:
```
conda env export --prefix /Users/daniel/Desktop/project_1/env > environment.yml
```

After running the export command, we can see our new `.yml` file stored as `environment.yml`.

A sample .yml file might look like the following:
```
name: my_ml_env
dependencies:
  - numpy
  - pandas
  - scikit-learn
  - jupyter
  - matplotlib
```

Of course, your actual file will depend on the packages you've installed in your environment.

For more on sharing an environment, check out the Conda documentation on sharing environments https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#sharing-an-environment.

Finally, to create an environment called env_from_file from a .yml file called environment.yml, you can run the command:
```
conda env create --file environment.yml --name env_from_file
```

For more on creating an environment from a .yml file, check out the Conda documentation on creating an environment from a .yml file https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file.


https://jupyter-notebook.readthedocs.io/en/stable/
https://www.dataquest.io/blog/jupyter-notebook-tutorial/

https://pandas.pydata.org/pandas-docs/stable/

https://github.com/mrdbourke/zero-to-mastery-ml/blob/master/section-2-data-science-and-ml-tools/introduction-to-pandas-video.ipynb

https://github.com/mrdbourke/zero-to-mastery-ml/blob/master/section-2-data-science-and-ml-tools/introduction-to-pandas.ipynb

https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html#min


https://numpy.org/doc/

https://github.com/mrdbourke/zero-to-mastery-ml/blob/master/section-2-data-science-and-ml-tools/introduction-to-numpy.ipynb

https://www.mathsisfun.com/data/standard-deviation.html

https://numpy.org/doc/stable/user/basics.broadcasting.html

https://jakevdp.github.io/PythonDataScienceHandbook/02.02-the-basics-of-numpy-arrays.html

http://jalammar.github.io/visual-numpy/

https://www.mathsisfun.com/algebra/matrix-multiplying.html

http://matrixmultiplication.xyz/

https://matplotlib.org/3.1.1/contents.html


### Scikit-learn
https://github.com/mrdbourke/zero-to-mastery-ml/blob/master/section-2-data-science-and-ml-tools/introduction-to-scikit-learn.ipynb

https://scikit-learn.org/stable/user_guide.html

https://github.com/mrdbourke/zero-to-mastery-ml/blob/master/section-2-data-science-and-ml-tools/scikit-learn-what-were-covering.ipynb

https://github.com/mrdbourke/zero-to-mastery-ml/blob/master/section-2-data-science-and-ml-tools/scikit-learn-workflow-example.ipynb

https://rahul-saini.medium.com/feature-scaling-why-it-is-required-8a93df1af310

https://benalexkeen.com/feature-scaling-with-scikit-learn/

https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/


https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

https://scikit-learn.org/0.15/modules/model_evaluation.html

#### Metrics
https://scikit-learn.org/stable/modules/model_evaluation.html

ROC and AUC https://www.youtube.com/watch?v=4jRBRDbJemM&ab_channel=StatQuestwithJoshStarmer

https://towardsdatascience.com/beyond-accuracy-precision-and-recall-3da06bea9f6c

https://stackoverflow.com/questions/17197492/is-there-a-library-function-for-root-mean-square-error-rmse-in-python/37861832#37861832

### Interviews
https://github.com/rohandm/data-science-interviews

https://github.com/mrdbourke/zero-to-mastery-ml/blob/master/section-2-data-science-and-ml-tools/introduction-to-scikit-learn-video.ipynb

https://github.com/mrdbourke/zero-to-mastery-ml/blob/master/section-2-data-science-and-ml-tools/introduction-to-scikit-learn.ipynb


https://github.com/mrdbourke/zero-to-mastery-ml/blob/master/section-2-data-science-and-ml-tools/scikit-learn-exercises.ipynb

https://github.com/mrdbourke/zero-to-mastery-ml/blob/master/section-2-data-science-and-ml-tools/scikit-learn-exercises-solutions.ipynb

UCI Machine Learning repository - https://archive.ics.uci.edu/ml/datasets/heart+disease
