import pandas as pd 
import numpy as np
import os 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.style.use('ggplot')

# Prepare data frame for overall summary
summaryTable = pd.DataFrame(columns=["Peak Training Accuracy","Peak Training Accuracy Epoch",
                                     "Lowest Training Loss","Lowest Training Loss Epoch",
                                     "Peak Validation Accuracy","Peak Validation Accuracy Epoch",
                                     "Lowest Validation Loss","Lowest Validation Loss Epoch",
                                     "Peak Training F1","Peak Training F1 Epoch",
                                     "Peak Validation F1","Peak Validation F1 Epoch",
                                     "Peak Training AUC", "Peak Training AUC Epoch",
                                     "Peak Validation AUC", "Peak Validation AUC Epoch",
                                     "Test Accuracy","Test Loss",
                                     "Class Test Loss", "Segmentation Test Loss",
                                     "Test F1", "Test AUC"])


def addToSummaryTable(experimentName='unknown', peakTrainingAccuracy=0, peakTrainingAccuracyEpoch=0, lowestTrainingLoss=0, lowestTrainingLossEpoch=0,
    peakValidationAccuracy=0, peakValidationAccuracyEpoch=0, lowestValidationLoss=0, lowestValidationLossEpoch=0, 
    peakTrainingF1=0, peakTrainingF1Epoch=0, peakValidationF1=0, peakValidationF1Epoch=0, peakTrainingAUC=0, peakTrainingAUCEpoch=0, 
    peakValidationAUC=0, peakValidationAUCEpoch=0, testAccuracy=0, testLoss=0, classTestLoss=0, segmentationTestLoss=0, testF1=0, testAUC=0):

    # Add entry to summary table
    summaryTable.loc[experimentName] = [peakTrainingAccuracy, peakTrainingAccuracyEpoch, lowestTrainingLoss, lowestTrainingLossEpoch,
    peakValidationAccuracy, peakValidationAccuracyEpoch, lowestValidationLoss, lowestValidationLossEpoch, 
    peakTrainingF1, peakTrainingF1Epoch, peakValidationF1, peakValidationF1Epoch, peakTrainingAUC, peakTrainingAUCEpoch, 
    peakValidationAUC, peakValidationAUCEpoch, testAccuracy, testLoss, classTestLoss, segmentationTestLoss, testF1, testAUC]
    

# Function to load data and tidy slightly - each run will add output to summary table automatically
def visualiseResults(experimentName, test=True, schema=3, type="", savefig=True):
    
    # Load in training/validation data and change column names and indexing
    df = pd.read_csv(experimentName+"/result_outputs/summary.csv") 
    
    if(schema==1):
        # Old columns
        df.columns = ['Training Accuracy','Training Loss','Validation Accuracy', 'Validation Loss', 'Epoch', 
                        'Data Loading Time', 'Computation Time']
    elif(schema==2):
        # Round 4 columns
        df.columns = ['Training Accuracy','Training Loss','Validation Accuracy', 
                      'Validation Loss', 'Class Training Loss', 'Segmentation Training Loss',
                      'Class Validation Loss','Segmentation Validation Loss', 
                      'Training F1', 'Validation F1',
                      'Epoch', 'Data Loading Time', 'Computation Time']
    elif(schema==3):
        # New columns
        df.columns = ['Training Accuracy','Training Loss','Validation Accuracy', 
                      'Validation Loss', 'Class Training Loss', 'Class Validation Loss',
                      'Segmentation Training Loss','Segmentation Validation Loss', 
                      'Training F1', 'Validation F1',
                      'Epoch', 'Data Loading Time', 'Computation Time','Training AUC','Validation AUC']

    if(test):
        # Load in test data
        test_df = pd.read_csv(experimentName+"/result_outputs/test_summary.csv") 

        if(schema==1):
            test_df.columns = ['Test Accuracy', 'Test Loss']
        else:
            test_df.columns = ['Test Accuracy', 'Test Loss', 'Class Test Loss', 'Segmentation Test Loss', 'Test F1', 'Test AUC']
    
    # Check for type of experiment
    if(type=='segmentation'):
        # Produce line graphs
        fig, ax = plt.subplots(ncols=2,figsize=(15,5))
        
        ax[0].plot('Epoch', 'Training Loss', data=df, marker='', color='red', linewidth=2, label='Training Loss')
        ax[0].plot('Epoch', 'Validation Loss', data=df, marker='', color='blue', linewidth=2, label='Validation Loss')
        ax[0].set_title("Loss of "+ experimentName + " Model")
        
        ax[1].plot('Epoch', 'Data Loading Time', data=df, marker='', color='red', linewidth=2, label='Data Loading Time')
        ax[1].plot('Epoch', 'Computation Time', data=df, marker='', color='blue', linewidth=2, label='Computation Time')
        ax[1].set_title("Timing of "+ experimentName + " Model")
        
        #ax[0].set_ylim(0.75,1)
        #ax[1].set_ylim(0.75,1)
        
        ax[0].legend()
        ax[1].legend()
    else:
        # Produce line graphs
        fig, ax = plt.subplots(ncols=3,figsize=(15,5))
        
        ax[0].plot('Epoch', 'Training Accuracy', data=df, marker='', color='red', linewidth=2, label='Training Accuracy')
        ax[0].plot('Epoch', 'Validation Accuracy', data=df, marker='', color='blue', linewidth=2, label='Validation Accuracy')
        ax[0].set_title("Accuracy of "+ experimentName + " Model")
        
        ax[1].plot('Epoch', 'Training Loss', data=df, marker='', color='red', linewidth=2, label='Training Loss')
        ax[1].plot('Epoch', 'Validation Loss', data=df, marker='', color='blue', linewidth=2, label='Validation Loss')
        ax[1].set_title("Loss of "+ experimentName + " Model")
        
        ax[2].plot('Epoch', 'Data Loading Time', data=df, marker='', color='red', linewidth=2, label='Data Loading Time')
        ax[2].plot('Epoch', 'Computation Time', data=df, marker='', color='blue', linewidth=2, label='Computation Time')
        ax[2].set_title("Timing of "+ experimentName + " Model")
        
        ax[0].set_ylim(0.75,1)
        #ax[1].set_ylim(0.75,1)
        
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
    
    # Calculate peak epochs and best results
    print("Peak Training Accuracy of "+str(round(df['Training Accuracy'].values.max(),4))+" was attained at epoch "+str(df['Training Accuracy'].values.argmax()))
    print("Peak Validation Accuracy of "+str(round(df['Validation Accuracy'].values.max(),4))+" was attained at epoch "+str(df['Validation Accuracy'].values.argmax()))
    print("\n")
    print("Lowest Training Loss of "+str(round(df['Training Loss'].values.min(),4))+" was attained at epoch "+str(df['Training Loss'].values.argmin()))
    print("Lowest Validation Loss of "+str(round(df['Validation Loss'].values.min(),4))+" was attained at epoch "+str(df['Validation Loss'].values.argmin()))
    print("\n")
    
    if(test):
        # Display Test Results
        print("Test Accuracy is "+str(round(test_df['Test Accuracy'].values.max(),4)))
        print("Test Loss is "+str(round(test_df['Test Loss'].values.min(),4)))

        if(schema==1):
            # Add entry to summary table
            addToSummaryTable(experimentName, df['Training Accuracy'].values.max(),df['Training Accuracy'].values.argmax(),
                                                df['Training Loss'].values.min(),df['Training Loss'].values.argmin(),
                                                df['Validation Accuracy'].values.max(),df['Validation Accuracy'].values.argmax(),
                                                df['Validation Loss'].values.min(),df['Validation Loss'].values.argmin(),
                                                testAccuracy=test_df['Test Accuracy'].values.max(),testLoss=test_df['Test Loss'].values.min())
        elif(schema==2):
            # Add entry to summary table
            addToSummaryTable(experimentName, df['Training Accuracy'].values.max(),df['Training Accuracy'].values.argmax(),
                                                df['Training Loss'].values.min(),df['Training Loss'].values.argmin(),
                                                df['Validation Accuracy'].values.max(),df['Validation Accuracy'].values.argmax(),
                                                df['Validation Loss'].values.min(),df['Validation Loss'].values.argmin(),
                                                peakTrainingF1=df['Training F1'].values.max(),peakTrainingF1Epoch=df['Training F1'].values.argmax(),
                                                peakValidationF1=df['Validation F1'].values.max(),peakValidationF1Epoch=df['Validation F1'].values.argmax(),
                                                testAccuracy=test_df['Test Accuracy'].values.max(),testLoss=test_df['Test Loss'].values.min())
                                                

        elif(schema==3):
            # Add entry to summary table
            addToSummaryTable(experimentName, df['Training Accuracy'].values.max(),df['Training Accuracy'].values.argmax(),
                                                df['Training Loss'].values.min(),df['Training Loss'].values.argmin(),
                                                df['Validation Accuracy'].values.max(),df['Validation Accuracy'].values.argmax(),
                                                df['Validation Loss'].values.min(),df['Validation Loss'].values.argmin(),
                                                peakTrainingF1=df['Training F1'].values.max(),peakTrainingF1Epoch=df['Training F1'].values.argmax(),
                                                peakValidationF1=df['Validation F1'].values.max(),peakValidationF1Epoch=df['Validation F1'].values.argmax(),
                                                peakTrainingAUC=df['Training AUC'].values.max(),peakTrainingAUCEpoch=df['Training AUC'].values.argmax(),
                                                peakValidationAUC=df['Validation AUC'].values.max(),peakValidationAUCEpoch=df['Validation AUC'].values.argmax(),
                                                testAccuracy=test_df['Test Accuracy'].values.max(),testLoss=test_df['Test Loss'].values.min())
    else:
        if(schema==1):
            # Add entry to summary table
            addToSummaryTable(experimentName, df['Training Accuracy'].values.max(),df['Training Accuracy'].values.argmax(),
                                                df['Training Loss'].values.min(),df['Training Loss'].values.argmin(),
                                                df['Validation Accuracy'].values.max(),df['Validation Accuracy'].values.argmax(),
                                                df['Validation Loss'].values.min(),df['Validation Loss'].values.argmin())
        elif(schema==2):
            # Add entry to summary table
            addToSummaryTable(experimentName, df['Training Accuracy'].values.max(),df['Training Accuracy'].values.argmax(),
                                                df['Training Loss'].values.min(),df['Training Loss'].values.argmin(),
                                                df['Validation Accuracy'].values.max(),df['Validation Accuracy'].values.argmax(),
                                                df['Validation Loss'].values.min(),df['Validation Loss'].values.argmin(),
                                                peakTrainingF1=df['Training F1'].values.max(),peakTrainingF1Epoch=df['Training F1'].values.argmax(),
                                                peakValidationF1=df['Validation F1'].values.max(),peakValidationF1Epoch=df['Validation F1'].values.argmax())
        elif(schema==3):
            # Add entry to summary table
            addToSummaryTable(experimentName, df['Training Accuracy'].values.max(),df['Training Accuracy'].values.argmax(),
                                                df['Training Loss'].values.min(),df['Training Loss'].values.argmin(),
                                                df['Validation Accuracy'].values.max(),df['Validation Accuracy'].values.argmax(),
                                                df['Validation Loss'].values.min(),df['Validation Loss'].values.argmin(),
                                                peakTrainingF1=df['Training F1'].values.max(),peakTrainingF1Epoch=df['Training F1'].values.argmax(),
                                                peakValidationF1=df['Validation F1'].values.max(),peakValidationF1Epoch=df['Validation F1'].values.argmax(),
                                                peakTrainingAUC=df['Training AUC'].values.max(),peakTrainingAUCEpoch=df['Training AUC'].values.argmax(),
                                                peakValidationAUC=df['Validation AUC'].values.max(),peakValidationAUCEpoch=df['Validation AUC'].values.argmax())
    if(savefig):
        fig.savefig(experimentName+".pdf")

    return(df)

# Function to load data and tidy slightly - each run will add output to summary table automatically
# Handles a list of experiments to see side by side

def visualiseMergedResults(experimentNames, numberOfSeeds, test=True, schema=3, includeAllExperiments=False, startSeedIdx=0, type='', savefig=True):
      
    # Produce line graph grid to match merged experiments
    if(schema==3):
        if(type=='segmentation'):
            nrows = len(experimentNames)
        else:
            nrows = 4*len(experimentNames)
    elif(schema==2):
        nrows = 3*len(experimentNames)
    else:
        nrows = 2*len(experimentNames)

    fig, ax = plt.subplots(nrows = nrows, ncols=numberOfSeeds+1,figsize=(15*(numberOfSeeds+1),10*nrows))
    
    for idx, experimentName in enumerate(experimentNames):
        
        # Reset cumulative df
        cumulativedf = pd.DataFrame()
        cumulativetestdf = pd.DataFrame()
        
        for seed in range(startSeedIdx, numberOfSeeds + startSeedIdx):
    
            # Load in training/validation data and change column names and indexing
            df = pd.read_csv(experimentName+"_s"+str(seed)+"/result_outputs/summary.csv") 
    
            if(schema==1):
                # Old columns
                df.columns = ['Training Accuracy','Training Loss','Validation Accuracy', 'Validation Loss', 'Epoch', 'Data Loading Time', 'Computation Time']
            elif(schema==2):
                # New columns
                df.columns = ['Training Accuracy','Training Loss','Validation Accuracy', 
                              'Validation Loss', 'Class Training Loss', 'Segmentation Training Loss',
                              'Class Validation Loss','Segmentation Validation Loss', 
                              'Training F1', 'Validation F1',
                              'Epoch', 'Data Loading Time', 'Computation Time']
            elif(schema==3):
                # New columns
                df.columns = ['Training Accuracy','Training Loss','Validation Accuracy', 
                              'Validation Loss', 'Class Training Loss', 'Class Validation Loss',
                              'Segmentation Training Loss','Segmentation Validation Loss', 
                              'Training F1', 'Validation F1',
                              'Epoch', 'Data Loading Time', 'Computation Time','Training AUC','Validation AUC']

            if(test):
                # Load in test data
                test_df = pd.read_csv(experimentName+"_s"+str(seed)+"/result_outputs/test_summary.csv") 

                if(schema==1):
                    test_df.columns = ['Test Accuracy', 'Test Loss']
                else:
                    test_df.columns = ['Test Accuracy', 'Test Loss', 'Class Test Loss', 'Segmentation Test Loss', 'Test F1', 'Test AUC']

            if(type=='segmentation'):
                if(len(experimentNames)==1):                  
                    ax[seed - startSeedIdx].plot('Epoch', 'Segmentation Training Loss', data=df, marker='', color='red', linewidth=2, label='Training Loss')
                    ax[seed - startSeedIdx].plot('Epoch', 'Segmentation Validation Loss', data=df, marker='', color='blue', linewidth=2, label='Validation Loss')
                    ax[seed - startSeedIdx].set_title("Loss of "+ experimentName + " (Seed="+str(seed)+") Model")

                    # Graph limits
                    #ax[idx, seed - startSeedIdx].set_ylim(0.75,1)

                    ax[seed - startSeedIdx].legend()
                else:
                    ax[idx, seed - startSeedIdx].plot('Epoch', 'Segmentation Training Loss', data=df, marker='', color='red', linewidth=2, label='Training Loss')
                    ax[idx, seed - startSeedIdx].plot('Epoch', 'Segmentation Validation Loss', data=df, marker='', color='blue', linewidth=2, label='Validation Loss')
                    ax[idx, seed - startSeedIdx].set_title("Loss of "+ experimentName + " (Seed="+str(seed)+") Model")
                
                    ax[idx, seed - startSeedIdx].legend()
            else:
                ax[idx, seed - startSeedIdx].plot('Epoch', 'Training Accuracy', data=df, marker='', color='red', linewidth=2, label='Training Accuracy')
                ax[idx, seed - startSeedIdx].plot('Epoch', 'Validation Accuracy', data=df, marker='', color='blue', linewidth=2, label='Validation Accuracy')
                ax[idx, seed - startSeedIdx].set_title("Accuracy of "+ experimentName + " (Seed="+str(seed)+") Model")

                ax[idx+len(experimentNames), seed - startSeedIdx].plot('Epoch', 'Training Loss', data=df, marker='', color='red', linewidth=2, label='Training Loss')
                ax[idx+len(experimentNames), seed - startSeedIdx].plot('Epoch', 'Validation Loss', data=df, marker='', color='blue', linewidth=2, label='Validation Loss')
                ax[idx+len(experimentNames), seed - startSeedIdx].set_title("Loss of "+ experimentName + " (Seed="+str(seed)+") Model")

                if(schema==3 or schema==2):
                    ax[idx+(2*len(experimentNames)), seed - startSeedIdx].plot('Epoch', 'Training F1', data=df, marker='', color='red', linewidth=2, label='Training F1')
                    ax[idx+(2*len(experimentNames)), seed - startSeedIdx].plot('Epoch', 'Validation F1', data=df, marker='', color='blue', linewidth=2, label='Validation F1')
                    ax[idx+(2*len(experimentNames)), seed - startSeedIdx].set_title("F1 of "+ experimentName + " (Seed="+str(seed)+") Model")
                    ax[idx+(2*len(experimentNames)), seed - startSeedIdx].legend()


                if(schema==3):
                    ax[idx+(3*len(experimentNames)), seed - startSeedIdx].plot('Epoch', 'Training AUC', data=df, marker='', color='red', linewidth=2, label='Training AUC')
                    ax[idx+(3*len(experimentNames)), seed - startSeedIdx].plot('Epoch', 'Validation AUC', data=df, marker='', color='blue', linewidth=2, label='Validation AUC')
                    ax[idx+(3*len(experimentNames)), seed - startSeedIdx].set_title("AUC of "+ experimentName + " (Seed="+str(seed)+") Model")    
                    ax[idx+(3*len(experimentNames)), seed - startSeedIdx].legend()

                # Graph limits
                ax[idx, seed - startSeedIdx].set_ylim(0.75,1)
                #ax[idx+len(experimentNames), seed].set_ylim(0.75,1)
                #ax[idx+(2*len(experimentNames)), seed].set_ylim(0.75,1)
                
                ax[idx, seed - startSeedIdx].legend()
                ax[idx+len(experimentNames), seed - startSeedIdx].legend()
            
            cumulativedf = pd.concat([cumulativedf, df], axis=1, join_axes=[df.index])
            
            if(test):
                cumulativetestdf = pd.concat([cumulativetestdf, test_df], axis=1, join_axes=[test_df.index])

            if(includeAllExperiments):
                if(test):
                    if(schema==1):
                        # Add entry to summary table
                        addToSummaryTable(experimentName, df['Training Accuracy'].values.max(),df['Training Accuracy'].values.argmax(),
                                                            df['Training Loss'].values.min(),df['Training Loss'].values.argmin(),
                                                            df['Validation Accuracy'].values.max(),df['Validation Accuracy'].values.argmax(),
                                                            df['Validation Loss'].values.min(),df['Validation Loss'].values.argmin(),
                                                            testAccuracy=test_df['Test Accuracy'].values.max(),testLoss=test_df['Test Loss'].values.min())
                    elif(schema==2):
                        # Add entry to summary table
                        addToSummaryTable(experimentName, df['Training Accuracy'].values.max(),df['Training Accuracy'].values.argmax(),
                                                            df['Training Loss'].values.min(),df['Training Loss'].values.argmin(),
                                                            df['Validation Accuracy'].values.max(),df['Validation Accuracy'].values.argmax(),
                                                            df['Validation Loss'].values.min(),df['Validation Loss'].values.argmin(),
                                                            peakTrainingF1=df['Training F1'].values.max(),peakTrainingF1Epoch=df['Training F1'].values.argmax(),
                                                            peakValidationF1=df['Validation F1'].values.max(),peakValidationF1Epoch=df['Validation F1'].values.argmax(),
                                                            testAccuracy=test_df['Test Accuracy'].values.max(),testLoss=test_df['Test Loss'].values.min())
                    elif(schema==3):
                        # Add entry to summary table
                        addToSummaryTable(experimentName, df['Training Accuracy'].values.max(),df['Training Accuracy'].values.argmax(),
                                                df['Training Loss'].values.min(),df['Training Loss'].values.argmin(),
                                                df['Validation Accuracy'].values.max(),df['Validation Accuracy'].values.argmax(),
                                                df['Validation Loss'].values.min(),df['Validation Loss'].values.argmin(),
                                                peakTrainingF1=df['Training F1'].values.max(),peakTrainingF1Epoch=df['Training F1'].values.argmax(),
                                                peakValidationF1=df['Validation F1'].values.max(),peakValidationF1Epoch=df['Validation F1'].values.argmax(),
                                                peakTrainingAUC=df['Training AUC'].values.max(),peakTrainingAUCEpoch=df['Training AUC'].values.argmax(),
                                                peakValidationAUC=df['Validation AUC'].values.max(),peakValidationAUCEpoch=df['Validation AUC'].values.argmax(),
                                                testAccuracy=test_df['Test Accuracy'].values.max(),testLoss=test_df['Test Loss'].values.min())
                else:
                    if(schema==1):
                        # Add entry to summary table
                        addToSummaryTable(experimentName, df['Training Accuracy'].values.max(),df['Training Accuracy'].values.argmax(),
                                                            df['Training Loss'].values.min(),df['Training Loss'].values.argmin(),
                                                            df['Validation Accuracy'].values.max(),df['Validation Accuracy'].values.argmax(),
                                                            df['Validation Loss'].values.min(),df['Validation Loss'].values.argmin())
                    elif(schema==2):
                        # Add entry to summary table
                        addToSummaryTable(experimentName, df['Training Accuracy'].values.max(),df['Training Accuracy'].values.argmax(),
                                                            df['Training Loss'].values.min(),df['Training Loss'].values.argmin(),
                                                            df['Validation Accuracy'].values.max(),df['Validation Accuracy'].values.argmax(),
                                                            df['Validation Loss'].values.min(),df['Validation Loss'].values.argmin(),
                                                            peakTrainingF1=df['Training F1'].values.max(),peakTrainingF1Epoch=df['Training F1'].values.argmax(),
                                                            peakValidationF1=df['Validation F1'].values.max(),peakValidationF1Epoch=df['Validation F1'].values.argmax())
                    elif(schema==3):
                        # Add entry to summary table
                        addToSummaryTable(experimentName, df['Training Accuracy'].values.max(),df['Training Accuracy'].values.argmax(),
                                                df['Training Loss'].values.min(),df['Training Loss'].values.argmin(),
                                                df['Validation Accuracy'].values.max(),df['Validation Accuracy'].values.argmax(),
                                                df['Validation Loss'].values.min(),df['Validation Loss'].values.argmin(),
                                                peakTrainingF1=df['Training F1'].values.max(),peakTrainingF1Epoch=df['Training F1'].values.argmax(),
                                                peakValidationF1=df['Validation F1'].values.max(),peakValidationF1Epoch=df['Validation F1'].values.argmax(),
                                                peakTrainingAUC=df['Training AUC'].values.max(),peakTrainingAUCEpoch=df['Training AUC'].values.argmax(),
                                                peakValidationAUC=df['Validation AUC'].values.max(),peakValidationAUCEpoch=df['Validation AUC'].values.argmax())
            
    # Summarise across seeds
    cumulativedf['Training Accuracy Average'] = cumulativedf['Training Accuracy'].mean(axis=1)
    cumulativedf['Training Accuracy Std'] = cumulativedf['Training Accuracy'].std(axis=1)
    cumulativedf['Training Loss Average'] = cumulativedf['Training Loss'].mean(axis=1)
    cumulativedf['Training Loss Std'] = cumulativedf['Training Loss'].std(axis=1)
    cumulativedf['Validation Accuracy Average'] = cumulativedf['Validation Accuracy'].mean(axis=1)
    cumulativedf['Validation Accuracy Std'] = cumulativedf['Validation Accuracy'].std(axis=1)
    cumulativedf['Validation Loss Average'] = cumulativedf['Validation Loss'].mean(axis=1)
    cumulativedf['Validation Loss Std'] = cumulativedf['Validation Loss'].std(axis=1)
    if(schema==2 or schema==3):
        cumulativedf['Training F1 Average'] = cumulativedf['Training F1'].mean(axis=1)
        cumulativedf['Training F1 Std'] = cumulativedf['Training F1'].std(axis=1)
        cumulativedf['Validation F1 Average'] = cumulativedf['Validation F1'].mean(axis=1)
        cumulativedf['Validation F1 Std'] = cumulativedf['Validation F1'].std(axis=1)
    if(schema==3):
        cumulativedf['Training AUC Average'] = cumulativedf['Training AUC'].mean(axis=1)
        cumulativedf['Training AUC Std'] = cumulativedf['Training AUC'].std(axis=1)
        cumulativedf['Validation AUC Average'] = cumulativedf['Validation AUC'].mean(axis=1)
        cumulativedf['Validation AUC Std'] = cumulativedf['Validation AUC'].std(axis=1)
        cumulativedf['Segmentation Validation Loss Average'] = cumulativedf['Segmentation Validation Loss'].mean(axis=1)
        cumulativedf['Segmentation Validation Loss Std'] = cumulativedf['Segmentation Validation Loss'].std(axis=1)
        cumulativedf['Segmentation Training Loss Average'] = cumulativedf['Segmentation Training Loss'].mean(axis=1)
        cumulativedf['Segmentation Training Loss Std'] = cumulativedf['Segmentation Training Loss'].std(axis=1)
    
    if(test):
        cumulativetestdf['Test Accuracy Average'] = cumulativetestdf['Test Accuracy'].mean(axis=1)
        cumulativetestdf['Test Loss Average'] = cumulativetestdf['Test Loss'].mean(axis=1)
        cumulativetestdf['Test Accuracy Std'] = cumulativetestdf['Test Accuracy'].std(axis=1)
        cumulativetestdf['Test Loss Std'] = cumulativetestdf['Test Loss'].std(axis=1)

        if(schema==2 or schema==3):
            cumulativetestdf['Class Test Loss Average'] = cumulativetestdf['Class Test Loss'].mean(axis=1)
            cumulativetestdf['Segmentation Test Loss Average'] = cumulativetestdf['Segmentation Test Loss'].mean(axis=1)
            cumulativetestdf['Class Test Loss Std'] = cumulativetestdf['Class Test Loss'].std(axis=1)
            cumulativetestdf['Segmentation Test Loss Std'] = cumulativetestdf['Segmentation Test Loss'].std(axis=1)
            cumulativetestdf['Test F1 Average'] = cumulativetestdf['Test F1'].mean(axis=1)
            cumulativetestdf['Test F1 Std'] = cumulativetestdf['Test F1'].std(axis=1)
            cumulativetestdf['Test AUC Average'] = cumulativetestdf['Test AUC'].mean(axis=1)
            cumulativetestdf['Test AUC Std'] = cumulativetestdf['Test AUC'].std(axis=1)

    if(type=='segmentation'):
        if(len(experimentNames)==1):
            ax[numberOfSeeds].plot('Epoch', 'Segmentation Training Loss Average', data=cumulativedf, marker='', color='red', linewidth=2, label='Training Accuracy')
            ax[numberOfSeeds].plot('Epoch', 'Segmentation Validation Loss Average', data=cumulativedf, marker='', color='blue', linewidth=2, label='Validation Accuracy')
            ax[numberOfSeeds].set_title("Average Segmentation Validation Loss Learning Curves of Best Multitask Model")

            # Specific code to get output for report
            legend_elements = [Line2D([0], [0], color='r', lw=4, label='Training Loss'),
                   Line2D([0], [0], color='b', label='Validation Loss')]
            ax[numberOfSeeds].legend(handles=legend_elements)
            ax[numberOfSeeds].set_xlabel("Epoch")
            ax[numberOfSeeds].set_ylabel("Segmentation Loss")
        else:
            ax[idx, numberOfSeeds].plot('Epoch', 'Segmentation Training Loss Average', data=cumulativedf, marker='', color='red', linewidth=2, label='Training Accuracy')
            ax[idx, numberOfSeeds].plot('Epoch', 'Segmentation Validation Loss Average', data=cumulativedf, marker='', color='blue', linewidth=2, label='Validation Accuracy')
            ax[idx, numberOfSeeds].set_title("Loss Average of "+ experimentName +" Model")
    else:
        # Make graphs for average traces
        ax[idx, numberOfSeeds].plot('Epoch', 'Training Accuracy Average', data=cumulativedf, marker='', color='red', linewidth=2, label='Training Accuracy')
        ax[idx, numberOfSeeds].plot('Epoch', 'Validation Accuracy Average', data=cumulativedf, marker='', color='blue', linewidth=2, label='Validation Accuracy')
        ax[idx, numberOfSeeds].set_title("Accuracy Average of "+ experimentName +" Model")
        
        ax[idx+len(experimentNames), numberOfSeeds].plot('Epoch', 'Training Loss Average', data=cumulativedf, marker='', color='red', linewidth=2, label='Training Accuracy')
        ax[idx+len(experimentNames), numberOfSeeds].plot('Epoch', 'Validation Loss Average', data=cumulativedf, marker='', color='blue', linewidth=2, label='Validation Accuracy')
        ax[idx+len(experimentNames), numberOfSeeds].set_title("Loss Average of "+ experimentName +" Model")

        if(schema==3 or schema==2):
            ax[idx+(2*len(experimentNames)), numberOfSeeds].plot('Epoch', 'Training F1 Average', data=cumulativedf, marker='', color='red', linewidth=2, label='Training F1')
            ax[idx+(2*len(experimentNames)), numberOfSeeds].plot('Epoch', 'Validation F1 Average', data=cumulativedf, marker='', color='blue', linewidth=2, label='Validation F1')
            ax[idx+(2*len(experimentNames)), numberOfSeeds].set_title("F1 Average of "+ experimentName + " (Seed="+str(seed)+") Model")
            ax[idx+(2*len(experimentNames)), numberOfSeeds].legend()
        if(schema==3):
            ax[idx+(3*len(experimentNames)), numberOfSeeds].plot('Epoch', 'Training AUC Average', data=cumulativedf, marker='', color='red', linewidth=2, label='Training AUC')
            ax[idx+(3*len(experimentNames)), numberOfSeeds].plot('Epoch', 'Validation AUC Average', data=cumulativedf, marker='', color='blue', linewidth=2, label='Validation AUC')
            ax[idx+(3*len(experimentNames)), numberOfSeeds].set_title("AUC Average of "+ experimentName + " (Seed="+str(seed)+") Model")

            # Specific code to get output for report
            legend_elements = [Line2D([0], [0], color='r', lw=4, label='Training AUC'),
                   Line2D([0], [0], color='b', label='Validation AUC')]
            ax[idx+(3*len(experimentNames)), numberOfSeeds].set_title("Average ROC-AUC Learning Curves of Best Multitask Model")
            ax[idx+(3*len(experimentNames)), numberOfSeeds].legend(handles=legend_elements)
            ax[idx+(3*len(experimentNames)), numberOfSeeds].set_xlabel("Epoch")
            ax[idx+(3*len(experimentNames)), numberOfSeeds].set_ylabel("ROC-AUC")

        # Set graphical limits
        ax[idx, numberOfSeeds].set_ylim(0.75,1)
        #ax[idx+len(experimentNames), seed].set_ylim(0.75,1)
        #ax[idx+(2*len(experimentNames)), seed].set_ylim(0.75,1)
        ax[idx, numberOfSeeds].legend()
        ax[idx+len(experimentNames), numberOfSeeds].legend()


    if(test):
    # Display Test Results

        if(schema==1):
            # Add entry to summary table
            addToSummaryTable("averaged_"+experimentName, cumulativedf['Training Accuracy Average'].values.max(),cumulativedf['Training Accuracy Average'].values.argmax(),
                                                cumulativedf['Training Loss'].values.min(),cumulativedf['Training Loss Average'].values.argmin(),
                                                cumulativedf['Validation Accuracy Average'].values.max(),cumulativedf['Validation Accuracy Average'].values.argmax(),
                                                cumulativedf['Validation Loss Average'].values.min(),cumulativedf['Validation Loss Average'].values.argmin(),
                                                testAccuracy=cumulativetestdf['Test Accuracy Average'].values.max(),testLoss=cumulativetestdf['Test Loss Average'].values.min())
        elif(schema==2):
            # Add entry to summary table
            addToSummaryTable("averaged_"+experimentName, cumulativedf['Training Accuracy Average'].values.max(),cumulativedf['Training Accuracy Average'].values.argmax(),
                                                cumulativedf['Training Loss'].values.min(),cumulativedf['Training Loss Average'].values.argmin(),
                                                cumulativedf['Validation Accuracy Average'].values.max(),cumulativedf['Validation Accuracy Average'].values.argmax(),
                                                cumulativedf['Validation Loss Average'].values.min(),cumulativedf['Validation Loss Average'].values.argmin(),
                                                peakTrainingF1=cumulativedf['Training F1 Average'].values.max(),peakTrainingF1Epoch=cumulativedf['Training F1 Average'].values.argmax(),
                                                peakValidationF1=cumulativedf['Validation F1 Average'].values.max(),peakValidationF1Epoch=cumulativedf['Validation F1 Average'].values.argmax(),
                                                testAccuracy=cumulativetestdf['Test Accuracy Average'].values.max(),testLoss=cumulativetestdf['Test Loss Average'].values.min())

        elif(schema==3):
            # Add entry to summary table
            addToSummaryTable("averaged_"+experimentName, cumulativedf['Training Accuracy Average'].values.max(),cumulativedf['Training Accuracy Average'].values.argmax(),
                                                cumulativedf['Training Loss'].values.min(),cumulativedf['Training Loss Average'].values.argmin(),
                                                cumulativedf['Validation Accuracy Average'].values.max(),cumulativedf['Validation Accuracy Average'].values.argmax(),
                                                cumulativedf['Validation Loss Average'].values.min(),cumulativedf['Validation Loss Average'].values.argmin(),
                                                peakTrainingF1=cumulativedf['Training F1 Average'].values.max(),peakTrainingF1Epoch=cumulativedf['Training F1 Average'].values.argmax(),
                                                peakValidationF1=cumulativedf['Validation F1 Average'].values.max(),peakValidationF1Epoch=cumulativedf['Validation F1 Average'].values.argmax(),
                                                testAccuracy=cumulativetestdf['Test Accuracy Average'].values.max(),testLoss=cumulativetestdf['Test Loss Average'].values.min(),
                                                peakTrainingAUC=cumulativedf['Training AUC Average'].values.max(),peakTrainingAUCEpoch=cumulativedf['Training AUC Average'].values.argmax(),
                                                peakValidationAUC=cumulativedf['Validation AUC Average'].values.max(),peakValidationAUCEpoch=cumulativedf['Validation AUC Average'].values.argmax(),
                                                classTestLoss=cumulativetestdf['Class Test Loss Average'].values.max(),segmentationTestLoss=cumulativetestdf['Segmentation Test Loss Average'].values.max(),
                                                testF1=cumulativetestdf['Test F1 Average'].values.max(),testAUC=cumulativetestdf['Test AUC Average'].values.max())
    else:
        if(schema==1):
            # Add entry to summary table
            addToSummaryTable("averaged_"+experimentName, cumulativedf['Training Accuracy Average'].values.max(),cumulativedf['Training Accuracy Average'].values.argmax(),
                                                cumulativedf['Training Loss'].values.min(),cumulativedf['Training Loss Average'].values.argmin(),
                                                cumulativedf['Validation Accuracy Average'].values.max(),cumulativedf['Validation Accuracy Average'].values.argmax(),
                                                cumulativedf['Validation Loss Average'].values.min(),cumulativedf['Validation Loss Average'].values.argmin())
        elif(schema==2):
            # Add entry to summary table
            addToSummaryTable("averaged_"+experimentName, cumulativedf['Training Accuracy Average'].values.max(),cumulativedf['Training Accuracy Average'].values.argmax(),
                                                cumulativedf['Training Loss'].values.min(),cumulativedf['Training Loss Average'].values.argmin(),
                                                cumulativedf['Validation Accuracy Average'].values.max(),cumulativedf['Validation Accuracy Average'].values.argmax(),
                                                cumulativedf['Validation Loss Average'].values.min(),cumulativedf['Validation Loss Average'].values.argmin(),
                                                peakTrainingF1=cumulativedf['Training F1 Average'].values.max(),peakTrainingF1Epoch=cumulativedf['Training F1 Average'].values.argmax(),
                                                peakValidationF1=cumulativedf['Validation F1 Average'].values.max(),peakValidationF1Epoch=cumulativedf['Validation F1 Average'].values.argmax())
        elif(schema==3):
                            # Add entry to summary table
            addToSummaryTable("averaged_"+experimentName, cumulativedf['Training Accuracy Average'].values.max(),cumulativedf['Training Accuracy Average'].values.argmax(),
                                                cumulativedf['Training Loss'].values.min(),cumulativedf['Training Loss Average'].values.argmin(),
                                                cumulativedf['Validation Accuracy Average'].values.max(),cumulativedf['Validation Accuracy Average'].values.argmax(),
                                                cumulativedf['Validation Loss Average'].values.min(),cumulativedf['Validation Loss Average'].values.argmin(),
                                                peakTrainingF1=cumulativedf['Training F1 Average'].values.max(),peakTrainingF1Epoch=cumulativedf['Training F1 Average'].values.argmax(),
                                                peakValidationF1=cumulativedf['Validation F1 Average'].values.max(),peakValidationF1Epoch=cumulativedf['Validation F1 Average'].values.argmax(),
                                                peakTrainingAUC=cumulativedf['Training AUC Average'].values.max(),peakTrainingAUCEpoch=cumulativedf['Training AUC Average'].values.argmax(),
                                                peakValidationAUC=cumulativedf['Validation AUC Average'].values.max(),peakValidationAUCEpoch=cumulativedf['Validation AUC Average'].values.argmax())
    if(savefig):
        fig.savefig(experimentNames[0]+".pdf")

    return cumulativetestdf
            