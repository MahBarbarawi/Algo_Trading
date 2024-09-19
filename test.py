import os.path

items= os.listdir(".\models\EURUSD\model")
print(items )
re = os.path.exists('.\models\EURUSD\model')
print(re)

re = os.makedirs(r'.\test-ee')
print(re)