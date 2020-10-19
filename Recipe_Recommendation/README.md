Built CNN models (Transfer Learning using Keras and TesorFlow) to classify food images with a 92% accuracy. Mapped food images to their recipe to provide recommendations based on ingredients and nutritional by computing text similarity amongst recipe clusters.


Steps to run the code:

1. Run all the cells in "Recipe_Classification_Model.ipynb" and generate a .hdf5 file of the model you would like to use for 
classification(or use the model file contained in the zip file)

2. Run all the cells in "Recipe_Clustering.ipynb". This file will use "simplified-recipes-1M.npz" file and "full_format_recipes.json" file to output a "processed_data.csv" file

3. Begin predicting your input image and get recommendatioons by loading the preferred model and set paths to images to be
classified in Recipe_Prediction_Recommendation. This file uses the model creaded from step 1 and data file created from step 2
