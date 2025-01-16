import numpy as np
import matplotlib.pyplot as plt

# Define loss functions
def mse_loss(y_true, y_pred):
    return 0.5 * (y_pred[1] - y_true[1])**2

def cce_loss(y_true, y_pred):
    return -y_true[1] * np.log(y_pred[1])

def dot_product_loss(y_true, y_pred):
    true_abs = np.sqrt(np.sum(y_true**2))
    pred_abs = np.sqrt(np.sum(y_pred**2))
    return 1 - np.sum((y_true * y_pred))/(true_abs * pred_abs)

def generalized_dice_loss(y_true, y_pred):

    epsilon = 1e-6  # Small value to avoid division by zero

    # Numerator and denominator for GDL
    numerator = 2 * np.sum(y_true * y_pred)
    denominator = np.sum((y_true + y_pred))


    return 1 - numerator / (denominator + epsilon)

# Prepare data
y_true = 1  # True label (change to 0 for the negative class)
y_pred_values = np.linspace(0, 1, 100)  # Predicted values from 0 to 1

# Compute losses
mse_losses = np.array([mse_loss(np.array([0,y_true]), np.array([1-y_pred,y_pred])) for y_pred in y_pred_values])
cce_losses = np.array([cce_loss(np.array([0,y_true]), np.array([1-y_pred,y_pred])) for y_pred in y_pred_values])
dot_losses = np.array([dot_product_loss(np.array([0,y_true]), np.array([1-y_pred,y_pred])) for y_pred in y_pred_values])
gdl_losses = np.array([generalized_dice_loss(np.array([0,y_true]), np.array([1-y_pred,y_pred])) for y_pred in y_pred_values])


# Plotting
plt.figure(figsize=(8, 8))
plt.plot(y_pred_values, mse_losses, label='MSE')
plt.plot(y_pred_values, cce_losses, label='CCE')
plt.plot(y_pred_values, dot_losses, label='DPL')
plt.plot(y_pred_values, gdl_losses, label='GDL')
# plt.title('', fontsize=16)
plt.xlabel('Predicted Value $\hat{y}$', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend(fontsize=12)
#plot major and minor grid lines
plt.grid(which='both')
#plot minor every 0.05, major every 0.1
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth='0.1', color='black')
plt.grid(which='major', linestyle='-', linewidth='0.3', color='black')
#change axis tick font size
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('loss_functions.png')
plt.tight_layout()
plt.show()
