# MNIST - Handwritten Digit Recognition Applications 

The goal of this project is to demonstrate a simple example of how to integrate a machine learning model into a real-life application. In order to do so, this repository holds the source code for a flask application integrated with a feature to predict whether a student will pass or fail a course given a specific set of inputs.

This project is to be used for introductory workshops aimed at teaching the basics of integrating ML into a Flask App.

## Prerequisites

Install necessary python packages

`pip install -r requirements.txt`

## Train the model 

Run `python -m model.model` from the **root** directory.

After training the model, the trained weights and the optimizers are saved in 

```bash
├── model
|    ├── results
|           ├── model.pth
|           ├── optimizer.pth
```

## Running the Flask Application 

Run the **app.py** file from the **root** directory. 

`python app.py` 

Go to **localhost:5000** to access the application from the browser of your choice.



