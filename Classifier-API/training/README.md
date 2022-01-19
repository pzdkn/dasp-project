# Training
## Short Intro
* run *train_and_evaluate.sh* for basic usage
## Add new classifcation and data type
* go to input.py and implement a new data processor following the same principles as the other ones. 
    Then update the dict of processors `processors`. This is later used to specifiy the correct processor
    with the `task_name` CLI argument in *train_and_evaluate.sh*
## New models
* trained new models are saved according to the naming convention. See under **Folder Structure** in the README in the parent folder
