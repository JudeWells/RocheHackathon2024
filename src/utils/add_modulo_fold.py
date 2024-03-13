import os
import pandas as pd



# Function to extract position and assign fold
def assign_fold(mutant):
    # Split the string by ':' in case of multiple positions
    positions = mutant.split(':')

    # Apply modulo operation to each position
    folds = [(int(''.join(filter(str.isdigit, pos))) % 5 if int(''.join(filter(str.isdigit, pos))) % 5 != 0 else 5) for
             pos in positions]

    # Join the results using '|'
    return '|'.join(map(str, folds))

def assign_all_folds(csv_dir):
    for fname in os.listdir(csv_dir):
        if fname.endswith('.csv'):
            df = pd.read_csv(os.path.join(csv_dir, fname))
            df['fold'] = df['mutant'].apply(assign_fold)
            df.to_csv(os.path.join(csv_dir, fname), index=False)
    # Apply the function to the 'mutant' column
    df['fold'] = df['mutant'].apply(assign_fold)

    # Save the updated dataframe to a new CSV file
    output_filename = 'path_to_output_file.csv'
    df.to_csv(output_filename, index=False)

if __name__== '__main__':
    csv_dir = '../DMS_datasets'
    assign_all_folds(csv_dir)
