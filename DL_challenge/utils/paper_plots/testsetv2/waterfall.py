import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

home = '/Users/mcgoug01/Cambridge University Dropbox/William McGough/AIvsRad/AI/predictions/fulldose_ROC/'
observations = home + 'observations.csv'
observations = pd.read_csv(observations)
#sort by confidence primarily, then by size if confidence is the same
observations = observations.sort_values(by=['Confidence','Size'],ascending=[True,True])
observations['Confidence'] -= 0.7017
#rescale confidence so that the smallest is -1 and the largest is 1
conf_min = observations['Confidence'].min()
conf_max = observations['Confidence'].max()
observations['Confidence'] = 2*(observations['Confidence']-conf_min)/(conf_max-conf_min)-1
highspec = 0.9999 - 0.705
highsens = 0.0442 - 0.705
new_highspec = 2*(highspec-conf_min)/(conf_max-conf_min)-1
new_highsens = 2*(highsens-conf_min)/(conf_max-conf_min)-1
# give new column according to new index
observations['alt_index'] = range(len(observations))
print(observations)

#where label is 0, set Size to 0
observations.loc[observations['Label']==0,'Size'] = 0

# find equivalent diameter of size, if size is volume and assumes it is a sphere
observations['Size'] = (6*observations['Size']/np.pi)**(1/3) / 10

#plot in waterfall style - a bar for each patient, with the height of the bar corresponding to the confidence at which the kidney goes from positive to negative
#colour the bar by the label of the kidney

# create two subplots - 2 rows, 1 column
fig, ax = plt.subplots(2,1,figsize=(10,10))
#plot smallest towards the left, largest towards the right - use alt_index to order
for i in observations.index:
    #if label is 1, colour red. elif SusMass is 1, colour black, elif cytic is 1, colour green, else colour blue
    if observations.loc[i,'Label'] == 1:
        ax[0].bar(observations.loc[i,'alt_index'],observations.loc[i,'Confidence'],color='red')
    elif observations.loc[i,'SusMass'] == 1:
        ax[0].bar(observations.loc[i,'alt_index'],observations.loc[i,'Confidence'],color='black')
    elif observations.loc[i,'Cystic'] == 1:
        ax[0].bar(observations.loc[i,'alt_index'],observations.loc[i,'Confidence'],color='green')
    else:
        ax[0].bar(observations.loc[i,'alt_index'],observations.loc[i,'Confidence'],color='blue')

#create a 4 bar legend for each of the colours
ax[0].bar(0,0,color='red',label='Cancer')
ax[0].bar(0,0,color='black',label='Healthy with Suspicious Mass')
ax[0].bar(0,0,color='green',label='Healthy with Cyst(s)')
ax[0].bar(0,0,color='blue',label='Healthy')

# plot the new highspec as a horizontal line with an annotation saying high specificity operating point
ax[0].axhline(y=new_highspec,color='black',linestyle='--')
ax[0].annotate('High Specificity Operating Point',xy=(0,new_highspec),xytext=(0.5,new_highspec-0.1),fontsize=10)
# same for high sensitivity - write annotation on the bottom right corner of the plot
ax[0].axhline(y=new_highsens,color='black',linestyle='--')
ax[0].annotate('High Sensitivity Operating Point',xy=(0,new_highsens),
               textcoords='axes fraction',xytext=(0.7,0.1),fontsize=10)
#turn off x axis labels
ax[0].set_xticks([])
#plot exact location of legend
ax[0].legend(loc='lower right', bbox_to_anchor=(1,0.2),fontsize=10)
# ax.set_xlabel('Patient')
ax[0].set_ylabel('Relative Confidence',fontdict={'fontsize':15})
#increase y tick font size
ax[0].tick_params(axis='y',labelsize=15)

#find alt_index where confidence is 0
zero_conf = observations['alt_index'][observations['Confidence']>0].values[0]

#draw a vertical line at this point in both subplots
ax[0].axvline(x=zero_conf,color='black',linestyle='--')
ax[1].axvline(x=zero_conf,color='black',linestyle='--')

#label line with 'Decision Boundary'
ax[1].annotate('Decision Boundary',xy=(zero_conf,5),xytext=(zero_conf-5,5),rotation=90,fontsize=10)

#reduce padding between subplots
plt.subplots_adjust(hspace=0.01)

# ax2.plot(observations['alt_index'],observations['Size'],'k--')
# plot sizes as a waterfall plot
for i in observations.index:
    ax[1].bar(observations.loc[i,'alt_index'],observations.loc[i,'Size'],color='black',alpha=0.5)

# fit logistic regression to the relationship between alt_index and diameter
x = observations['Confidence'][observations['Size']>0]
y = observations['Size'][observations['Size']>0]

#fit a logisttic regression to the data using the confidence as the independent variable and the size as the dependent variable
# this will be a continuous regression line
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import mean_squared_error
#
# x = x.values.reshape(-1,1)
# # apply inverse sigmoid to y using natural log
# # make y = 0.00001 where y = 0
# y[y==0] = 0.01
# print(y)
# print(y/(1-y))
# inv_y = np.log(y/(1-y))
# print(inv_y)

# #fit the logistic regression using size as a probability
# logreg = LogisticRegression()
# logreg.fit(x,inv_y)


# x = observations['alt_index']
# new_y = np.exp(logreg.predict(x.values.reshape(-1,1)))/(1+np.exp(logreg.predict(x.values.reshape(-1,1))))
# # if new_y is nan, set to 0
# new_y[np.isnan(new_y)] = 0

# ax[1].plot(x,new_y,'r--')

ax[1].set_xticks([])
ax[1].set_xlabel('Patient',fontdict={'fontsize':15})
ax[1].set_ylabel('Equivalent Diameter / cm',fontdict={'fontsize':15})
ax[1].tick_params(axis='y',labelsize=15)

#plot tight layout
# plt.grid(which='both')
# plt.minorticks_on()
# plt.grid(which='minor',axis='both',linestyle='--',linewidth=0.5)
# plt.grid(which='major',axis='both',linestyle='-',linewidth=1)

# apply the above to both subplots
for a in ax:
    a.grid(which='both')
    a.minorticks_on()
    a.grid(which='minor',axis='both',linestyle='--',linewidth=0.5)
    a.grid(which='major',axis='both',linestyle='-',linewidth=1)

#reduce padding between subplots and figure title
plt.tight_layout()

# add figure-wide title
# fig.suptitle(titlename,fontsize=15)
plt.savefig(home+'waterfall.png')
plt.savefig('/Users/mcgoug01/Cambridge University Dropbox/William McGough/nat_paper/waterfall_fulldose.png')

plt.show()



