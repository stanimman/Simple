# Source for the visualization
#https://www.kaggle.com/kanncaa1/pytorch-tutorial-for-deep-learning-lovers
loss_list = []
epoch_list = []
loss_list.append(epoch_loss)
epoch_list.append(epoch)
# visualization
plt.plot(epoch_list,loss_list)
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("Loss vs Number of Epoch")
plt.show()
