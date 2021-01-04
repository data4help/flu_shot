# flu_shot

This repository is our take on the DrivenData challenge Flu Shots (https://www.drivendata.org/competitions/66/flu-shot-learning/page/210/).
The code is split up into four pieces:

00_exploration:
Plotting the variables and taking a look which correlations with the targets could be meaningful to exploit

01_pipeline_processing:
Creation of a bunch of classes for the pre-processing. All classes are then lined up in a pipeline to be used
for testing and trainings data. Afterwards the processed data is saved in the folder "02_data" in order to not having
to process the data every time when trying a new model.

02_model_finder:
This file tries a bunch of different models on the data to find the one which is performing best. This cannot be done in a
class itself given that the next step would be to set the hyper-parameters to tune, which we don't know what they are, if
we don't know which model works best.

03_pipeline_predictions:
Here we create feature importance using a basic GradientBoosting model and grid-search to find the best hyper-parameters.
Additionally we introduce the SMOTE balancing method. All of that is again scripted into classes in order to use pipelines
			