# question 8

data = {
    '111': {'positive': 20, 'negative': 8},
    '100': {'positive': 20, 'negative': 17},
    '010': {'positive': 5, 'negative': 8},
    '000': {'positive': 5, 'negative': 17}
}

# Calculate the totals for independence checks
total_positives = sum(item['positive'] for item in data.values())
total_negatives = sum(item['negative'] for item in data.values())
total_instances = total_positives + total_negatives

# Calculate individual probabilities for X1 and X2
prob_X1_positive = (data['111']['positive'] + data['100']['positive']) / total_positives
prob_X2_positive = (data['111']['positive'] + data['010']['positive']) / total_positives
prob_X1_negative = (data['111']['negative'] + data['100']['negative']) / total_negatives
prob_X2_negative = (data['111']['negative'] + data['010']['negative']) / total_negatives

# Calculate joint probabilities for X1=1 and X2=1
joint_prob_positive = data['111']['positive'] / total_positives
joint_prob_negative = data['111']['negative'] / total_negatives

# Check for independence by comparing joint probabilities with the product of individual probabilities
independence_positive = joint_prob_positive == prob_X1_positive * prob_X2_positive
independence_negative = joint_prob_negative == prob_X1_negative * prob_X2_negative

# Check for conditional independence given the class (use Bayes' theorem for conditional probabilities)
# For positive class
cond_indep_positive = (joint_prob_positive * (total_positives / total_instances)) == \
                      (prob_X1_positive * prob_X2_positive * (total_positives / total_instances))

# For negative class
cond_indep_negative = (joint_prob_negative * (total_negatives / total_instances)) == \
                      (prob_X1_negative * prob_X2_negative * (total_negatives / total_instances))

# Output the results for independence and conditional independence
independence = {
    'independence_positive': independence_positive,
    'independence_negative': independence_negative,
    'conditional_independence_positive': cond_indep_positive,
    'conditional_independence_negative': cond_indep_negative
}

# Output the results for the class conditional probabilities required for naive Bayes prediction
class_conditional_probabilities = {
    'P(X1=1|positive)': prob_X1_positive,
    'P(X2=1|positive)': prob_X2_positive,
    'P(X1=1|negative)': prob_X1_negative,
    'P(X2=1|negative)': prob_X2_negative,
    'P(X3=1|positive)': data['111']['positive'] / total_positives,
    'P(X3=1|negative)': data['111']['negative'] / total_negatives
}

print(independence, class_conditional_probabilities)



P_X1_1_X2_1_positive = data['111']['positive'] / total_positives

# For negative class (number of negative examples where X1=1 and X2=1 divided by total number of negative examples)
P_X1_1_X2_1_negative = data['111']['negative'] / total_negatives

# Overall (number of examples where X1=1 and X2=1 divided by total number of examples)
P_X1_1_X2_1_overall = (data['111']['positive'] + data['111']['negative']) / total_instances

print(P_X1_1_X2_1_positive, P_X1_1_X2_1_negative, P_X1_1_X2_1_overall)



# To perform part (d) of question 8, we'll predict the class label for each example in the training set
# using the Naive Bayes classifier and compute the training error rate.

# To predict the class label for each example, we'll use the Naive Bayes formula which, for binary classification,
# compares the product of the likelihood of each feature and the class prior probabilities.
# We need the prior probabilities of the positive and negative classes:
prior_positive = total_positives / total_instances
prior_negative = total_negatives / total_instances

# For simplicity, let's assume binary features can only take on the values of 0 or 1
# and that the table represents all the possible combinations of feature values in the dataset.
# We'll calculate the likelihood for each feature combination given the class and multiply it by the prior
# and then normalize these values to get the posterior probabilities.
# We'll predict the class with the higher posterior probability and compare with the actual class.

# Calculate the likelihood of each feature combination given the class
likelihoods = {
    '111': {
        'positive': (prob_X1_positive * prob_X2_positive * class_conditional_probabilities['P(X3=1|positive)']),
        'negative': (prob_X1_negative * prob_X2_negative * class_conditional_probabilities['P(X3=1|negative)'])
    },
    '100': {
        'positive': (prob_X1_positive * (1 - prob_X2_positive) * (1 - class_conditional_probabilities['P(X3=1|positive)'])),
        'negative': (prob_X1_negative * (1 - prob_X2_negative) * (1 - class_conditional_probabilities['P(X3=1|negative)']))
    },
    '010': {
        'positive': ((1 - prob_X1_positive) * prob_X2_positive * (1 - class_conditional_probabilities['P(X3=1|positive)'])),
        'negative': ((1 - prob_X1_negative) * prob_X2_negative * (1 - class_conditional_probabilities['P(X3=1|negative)']))
    },
    '000': {
        'positive': ((1 - prob_X1_positive) * (1 - prob_X2_positive) * (1 - class_conditional_probabilities['P(X3=1|positive)'])),
        'negative': ((1 - prob_X1_negative) * (1 - prob_X2_negative) * (1 - class_conditional_probabilities['P(X3=1|negative)']))
    }
}

# Calculate the posterior probabilities and predictions
predictions = {}
errors = 0

for combination, counts in data.items():
    posterior_positive = likelihoods[combination]['positive'] * prior_positive
    posterior_negative = likelihoods[combination]['negative'] * prior_negative
    
    # Normalize posterior probabilities so they sum to 1
    total_posterior = posterior_positive + posterior_negative
    normalized_posterior_positive = posterior_positive / total_posterior
    normalized_posterior_negative = posterior_negative / total_posterior
    
    # The class prediction is the class with the higher posterior probability
    predicted_class = '+' if normalized_posterior_positive > normalized_posterior_negative else '-'
    actual_class = '+' if counts['positive'] > counts['negative'] else '-'
    
    # Record the prediction and whether it was correct or not
    predictions[combination] = {
        'predicted_class': predicted_class,
        'actual_class': actual_class,
        'correct': predicted_class == actual_class
    }
    
    # Count errors
    if predicted_class != actual_class:
        errors += counts['positive'] if actual_class == '+' else counts['negative']

# The training error rate is the total number of errors divided by the total number of instances
training_error_rate = errors / total_instances
print('-'*50)
print(predictions, training_error_rate)

print('-'*50)
# Question 10

# From the table provided, we extract the counts for A, B, C with Class '+' and A with Class '-'.
# We can then calculate the conditional probabilities.

# Table data provided by user
table_data = [
    {'A': 0, 'B': 0, 'C': 1, 'Class': '-'},
    {'A': 1, 'B': 0, 'C': 1, 'Class': '+'},
    {'A': 0, 'B': 1, 'C': 0, 'Class': '-'},
    {'A': 1, 'B': 0, 'C': 0, 'Class': '-'},
    {'A': 1, 'B': 0, 'C': 1, 'Class': '+'},
    {'A': 0, 'B': 0, 'C': 1, 'Class': '+'},
    {'A': 1, 'B': 1, 'C': 0, 'Class': '-'},
    {'A': 0, 'B': 0, 'C': 0, 'Class': '-'},
    {'A': 0, 'B': 1, 'C': 0, 'Class': '+'},
    {'A': 1, 'B': 1, 'C': 1, 'Class': '+'}
]

# Initialize counts
count_A_positive = count_B_positive = count_C_positive = count_A_negative = count_B_negative = count_C_negative = 0
total_positive = total_negative = 0

# Count the occurrences for each feature given the class
for instance in table_data:
    if instance['Class'] == '+':
        total_positive += 1
        if instance['A'] == 1:
            count_A_positive += 1
        if instance['B'] == 1:
            count_B_positive += 1
        if instance['C'] == 1:
            count_C_positive += 1
    elif instance['Class'] == '-':
        total_negative += 1
        if instance['A'] == 1:
            count_A_negative += 1
        if instance['B'] == 1:
            count_B_negative += 1
        if instance['C'] == 1:
            count_C_negative += 1
# Calculate the conditional probabilities
P_A_1_positive = count_A_positive / total_positive
P_B_1_positive = count_B_positive / total_positive
P_C_1_positive = count_C_positive / total_positive
P_A_1_negative = count_A_negative / total_negative
P_B_1_negative = count_B_negative / total_negative
P_C_1_negative = count_C_negative / total_negative

print(P_A_1_positive, P_B_1_positive, P_C_1_positive, P_A_1_negative, P_B_1_negative, P_C_1_negative, total_positive, total_negative)


likelihood_positive = P_A_1_positive * P_B_1_positive * P_C_1_positive

# Calculate the likelihood for the negative class
likelihood_negative = P_A_1_negative * P_B_1_negative * P_C_1_negative

# The class with the higher likelihood is the predicted class
predicted_class = '+' if likelihood_positive > likelihood_negative else '-'

prob_r = 0.1

likelihood_pos_given_class = (likelihood_positive * (total_positive / (total_positive + total_negative)))/prob_r

print(likelihood_positive, likelihood_negative, predicted_class, likelihood_pos_given_class)

# Manually counting from the provided data set for question 10 part (c):
# Counts for P(A=1), P(B=1), and P(A=1, B=1)

# Initialize counts
count_A_1 = count_B_1 = count_A_1_B_1 = 0

# Total number of instances
total_instances = len(table_data)

# Count occurrences for each case
for instance in table_data:
    if instance['A'] == 1:
        count_A_1 += 1
        if instance['B'] == 1:
            count_A_1_B_1 += 1
    if instance['B'] == 1:
        count_B_1 += 1

# Calculate probabilities
P_A_1 = count_A_1 / total_instances
P_B_1 = count_B_1 / total_instances
P_A_1_B_1 = count_A_1_B_1 / total_instances
independent ='yes' if (P_A_1_B_1 == P_A_1 * P_B_1) else 'no'
print(P_A_1, P_B_1, P_A_1_B_1, independent)


# Assuming uniform priors, since no prior information is given about the class distribution
prior_positive = 0.5
prior_negative = 0.5


#10 (b)
# Calculate the marginal probability of R
P_R = likelihood_positive * prior_positive + likelihood_negative * prior_negative

# Calculate the posterior probability P(+|R) using Bayes' theorem
P_positive_given_R = (likelihood_positive * prior_positive) / P_R

# Output the calculated probabilities
print(likelihood_positive, likelihood_negative, P_positive_given_R)
