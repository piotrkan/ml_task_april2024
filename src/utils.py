'''misc functions used in main'''
import pandas as pd
import matplotlib.pyplot as plt

def plot_train_val_curve(train_loss:list, val_loss:list, runname:str) -> None:
    '''function for plotting train-val curve while trianing neural network
    Args:
        train_loss - list with losses obtained during training
        val_loss - list with losses obtained during validation
        runname - runname, will be used while saving
    Out:
        None
    '''
    #assert same length
    assert len(train_loss)==len(val_loss)
    epoch_list = [i for i in range(len(train_loss))]
    
    #plot
    plt.plot(epoch_list, train_loss, 'blue', label='Training Loss')
    plt.plot(epoch_list, val_loss, 'orange', label='Validation Loss')
    plt.xlabel('No. epochs')
    plt.ylabel('MSE Loss')
    plt.suptitle('Train-val curve')
    plt.legend()
    plt.savefig(f'models/{runname}.png')
    plt.show()
    
def examine_seq(data:pd.DataFrame, col_of_interest:str)->set:
    '''function for exploring amino acids within a provided sequence in a dataframe,
        will examine aa proportions, lengths visually and textually
    
    Args:
        data - dataframe containing col_of_interest
        col_of_interest - column name containing sequence data
    Out:
        set of unique aminoacids present in the sequences (note: can contain non-aas)
    '''
    #check length of each sequence
    seq_length = data[col_of_interest].apply(lambda x: len(x))
    print('Unique lenghts of sequences in the df: ', set(seq_length),'\n')

    #check what aminoacids are present and in what amount
    all_aa = ''.join(data[col_of_interest])
    unique_aa= set(all_aa)
    print('Unique amino acids in the df: ',unique_aa)
    print('No of unique amino acids in the df: ', len(unique_aa),'\n')

    #check how frequent each aminoacid is in the sequences
    for aa in unique_aa:
        print(f'No. of sequences containing "{aa}": ',
              len(data.loc[data[col_of_interest].str.contains(aa)][col_of_interest]))

    #visualise
    plt.bar(list(unique_aa),[all_aa.count(aa)/len(all_aa) for aa in unique_aa])
    plt.ylabel('Proportion of AA across the sequence')
    plt.xlabel('Amino Acids Present')
    plt.show()
    return unique_aa

def clean_artifacts(data:pd.DataFrame, col_of_interest:str='sequence', artifact:str='-') -> pd.DataFrame:
    '''function for cleaning potential data artifacts present in the sequence column, will
    return cleaned data
    
    Args:
        data - dataframe containing col_of_interest
        col_of_interest - column name containing sequence data
        artifact - artifact to be cleaned
    Out:
        pd.DataFrame object without the artifact & paddings instead.
    '''
    #removing
    data[col_of_interest]=data[col_of_interest].str.replace(artifact,'')

    #check
    assert not any(data[col_of_interest].str.contains('-'))

    #zero padding
    max_seq_length = max([len(seq) for seq in data[col_of_interest].values])
    data[col_of_interest] = [seq +str(0)*(max_seq_length -len(seq)) for seq in data[col_of_interest].values]
    return data
