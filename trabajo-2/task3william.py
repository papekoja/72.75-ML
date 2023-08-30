import pandas as pd

# Replace 'data.csv' with the actual path to your CSV file
csv_file_path = 'trabajo-2/3-binary(TP1).csv'

# Read the CSV file into a DataFrame
data_df = pd.read_csv(csv_file_path)

# Calculate the prior probabilities
total_rows = len(data_df)
total_rank_1 = data_df[data_df['rank'] == 1].shape[0]
total_not_admitted = data_df[data_df['admit'] == 0].shape[0]
total_not_admitted_given_rank_1 = data_df[(data_df['rank'] == 1) & (data_df['admit'] == 0)].shape[0]

# Calculate probabilities
p_rank_1 = total_rank_1 / total_rows
p_not_admitted = total_not_admitted / total_rows
p_not_admitted_given_rank_1 = total_not_admitted_given_rank_1 / total_rank_1

# Apply Bayes' theorem
p_rank_1_given_not_admitted = (p_not_admitted_given_rank_1 * p_rank_1) / p_not_admitted

print(f'P(rank=1|not admitted) = {p_rank_1_given_not_admitted}')

# Calculate the conditional probability
total_rank_2_gre_450_gpa_35 = data_df[
    (data_df['rank'] == 2) &
    (data_df['gre'] < 500) &
    (data_df['gpa'] >= 3) &
    (data_df['admit'] == 1)
].shape[0]

total_rank_2_gre_450_gpa_35_total = data_df[
    (data_df['rank'] == 2) &
    (data_df['gre'] < 500) &
    (data_df['gpa'] >= 3)
].shape[0]

# Calculate the conditional probability
p_admitted_given_rank_2_gre_450_gpa_35 = total_rank_2_gre_450_gpa_35 / total_rank_2_gre_450_gpa_35_total

print("Probability of Admitted for Rank 2, GRE = 450, GPA = 3.5:", p_admitted_given_rank_2_gre_450_gpa_35)
