### Source code for the article [*Coordinating shiftable loads for collective photovoltaic self-consumption: a multi-agent approach* published](https://www.journals.elsevier.com/energy) in [Energy](https://www.journals.elsevier.com/energy)


oi



### Guidelines for using this repository

* Main Dependecies

  * Python ([SciPy](https://www.scipy.org/), [Pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/))
  * [Pyomo](http://www.pyomo.org/) optimization modeling language
  * It is suggested to use [Anaconda](https://www.anaconda.com) environments

#### Folders

  * The results used to produce the article results are in `\DATA\Results` as .mat files containing optimization solutions

  * `\DATA\PaperPics` contains the figures used in the article

  * `\DATA\Apps_List` contains the generated list of appliances characteristics

  * `\DATA\PV_sim.csv` is the PV production profile considered

#### Scripts

  * `AgentFunc.py` contains the [pyomo](http://www.pyomo.org/) models for centralized and MultiAgent setups (individual agents subproblems).

  * `MainScript.py` solves the models and implement the coordination algorithm

  * `PostProcess.py` contains functions to process the Pyomo outputs and generate .mat files

  * `Calculations.py` contains functions to produce a table with all the results

  * `SupportScript.py` generates plots

For a detailed explanation of the modelling process and formalism pelase refer to the paper available @ LINK

For questions please contact **Guilherme Luz** (`gpluz@fc.ul.pt`)
