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

# Function to view images of particular class from dataset

def viewimage(Dataset,class_label):
    """ Function to view first 10 images from Dataset with Image and label.	"""
    ct = 0
    fig = plt.figure(1, figsize=(16, 8))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 8), axes_pad=0.05)
    for i in range(len(transformed_train_dataset)):

        img, label = Dataset[i]
        if label == class_label :
          ct += 1
          #img2 = img.crop((100,0,280,512))
          print(label)
          ax = grid[ct]
          ax.imshow(np.asarray(img))
          if ct == 7 :
            break

            
import matplotlib.pyplot as plt
import numpy as np
learning_rate  = [1e-12,1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1 ]
log_learning_rate = np.log10(learning_rate)
train_loss = [0.7685,0.7671,0.7671,0.7064,0.5518,0.6689,0.7559]
plt.plot(log_learning_rate,train_loss)
plt.xlabel("Log Learning rate")
plt.ylabel("Train Loss")
plt.title("Train Loss vs Log Learning rate")
plt.show()
