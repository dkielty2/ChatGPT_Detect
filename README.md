# ChatGPT_Detect

## Using the Application
### Application Link
The current form of the application is available at this link:[http://makemake.wifi.cmu.edu/ChatGPT_Detect/application/](http://makemake.wifi.cmu.edu/ChatGPT_Detect/application/)

**Please note:** This is a temporary server so the application may not be accessible long-term after the completion of this project.

### Running the application locally
The application can alternatively be run on your local machine.
#### Additional Python dependencies
Beyond the default anaconda3 installation, the remaining required packages can be installed using:
```pip install torch torchvision contractions n2w```
#### Run
Once the additional dependencies are installed, the app can be run by entering the following into your terminal
```
$ cd application/
$ source source_this.sh
```
**Please note:** The first time this is run, multiple datasets from the natural language toolkit (nltk) and pytorch may be installed. This means startup time is significantly longer for the first run.

## Repository Layout
- application : code required to run the final application
- data : data used in this project
- essay_generator : scripts and notebooks used to query the Chat GPT API to construct the GPT part of the dataset
- metrics : exploration of different possible natural language processing metrics to be included in the model
- training : all work used for training the model




## Model Architecture
![diagram image](./diagram.png)



