import pickle
from utils import plot_acc_and_loss_all

with open(r'C:\Users\Cesc47\PycharmProjects\MCV\M4\Machine-Learning-for-Computer-Vision\week4\models\EfficientNetB2_exp_2_2022-01-29-20-23-42\saved_model\history_1.pickle', "rb") as input_file:
   history = pickle.load(input_file)

plot_acc_and_loss_all(history, r'C:\Users\Cesc47\PycharmProjects\MCV\M4\Machine-Learning-for-Computer-Vision\week4')