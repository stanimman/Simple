model_ft.eval()
pred_list = []
label_list = []
softmax_list = []
image_list = []
ct12 = 0 
for inputs1, labels1 in dataloaders['val']:
                ct12 += 1
                inputs1 = inputs1.to(device)
                labels1 = labels1.to(device)
                labels1 = labels1.type(torch.cuda.LongTensor)
                # zero the parameter gradients
                # optimizer.zero_grad()
                # forward
                # track history if only in train
                # with torch.set_grad_enabled(phase == 'train'):
                outputs1 = model_ft(inputs1)
                  
                    #print(outputs.shape)
                _, preds1 = torch.max(outputs1, 1)
                pred_list.extend(preds1.cpu().numpy())
                label_list.extend(labels1.data)
                #softmax_list.extend(softmax1(outputs1.cpu().detach().numpy(),axis =1))
                image_list.extend(inputs1)
                if ct12 == 50:
                  break
                  
def softmax1(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats. 
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the 
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter, 
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    
    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p
    
y_pred_prob = [i[1] for i in softmax_list]

!pip install pycm
from pycm import *
cm = ConfusionMatrix(actual_vector=np.asarray(label_list), predict_vector=np.asarray(pred_list))
#print(cm.classes)
#print(cm.table)
print(cm)

# calculate accuracy
from sklearn import metrics
print(metrics.roc_auc_score(label_list, y_pred_prob))
fpr, tpr, thresholds = metrics.roc_curve(label_list, y_pred_prob)

plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)

