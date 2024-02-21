**CS772 : Deep Learning for NLP : Assignment 1**

This is our attempt at the first assignment of the course CS772.  

To install all dependencies  
```pip install -r requirements.txt```

The learned weights of the final model are loaded in the form of a list (49 wts) in the UI. They can be found in `model_wts.pkl`

To boot up the User Interface for model inference    
```python ui.py```
```python3 ui.py```

The training loop of the main model can be found in `Palindrome.ipynb`, and the 4-fold cross validation and analysis is present in `Palindrome_Analysis.ipynb`. Note that the final model used for the interface is different from the one reported in 4-fold cross validation. This is because of the stochastic nature of weight initialization. The results are, however,  nearly same, since the hyperparameters and training methods used are identical.

The link to a video demonstration can be found at  
https://drive.google.com/file/d/1SI80BsDE57TNi9EU7wq3O5zObbCd66oi/view?usp=sharing  

A project by :-
1) Aziz Shameem : 20d070020  
2) Rohan Kalbaug : 20d170033  
3) Amruta Parulekar : 20d070009  
4) Keshav  Singhal : 20d070047