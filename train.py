from preporcess import prepare

data_dir = './datasets/Data_Train_Test'
model_dir = './results/' 
data_in = prepare(data_dir, cache=True)
