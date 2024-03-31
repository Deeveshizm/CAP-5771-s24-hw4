# Add import files
import pickle



# -----------------------------------------------------------
def question1():
    answers = {}

    # string "yes" or "no"
    answers["(a)"] = "no"
    answers["(b)"] = "no"
    answers["(c)"] = "yes" 
    answers["(d)"] = "yes"

    # explain-string: explanation in english prose
    answers["(a) explain"] = "No, because mutually exclusive means that if one rule is true none of the other rules can be true but here we can see overlaps in the rules"
    answers["(b) explain"] = "No, because there is no default rule and we know that exhaustive rules cover all the possibilities, hence if the there exists a scenario which is outside the coverage of these given set of rules the model will fail"
    answers["(c) explain"] = "Yes, eg. if the person is currently employed and has medium annual income the DB=no but if he is an home owner DB=yes always, so what happens when it's medium annual income, currently employed and he is also a home owner therefore ordering is needed."
    answers["(d) explain"] = "Yes, to make the rule set Exhaustive we need a default rule which which will help the rule set cover all the scenarios."

    return answers


# -----------------------------------------------------------
"""
def question2():
    answers = {}

    # string "yes" or "no"
    answers["(a)"] = None
    answers["(b)"] = None
    answers["(c)"] = None
    answers["(d)"] = None

    # explain_string: explanation in english prose
    answers["(a) explain"] = None
    answers["(b) explain"] = None
    answers["(c) explain"] = None
    answers["(d) explain"] = None

    return answers
"""
# -----------------------------------------------------------
def question3():
    answers = {}

    # string "yes" or "no"
    answers["(a)"] = 'yes' 
    answers["(b)"] = 'no'
    answers["(c)"] = 'no'

    # explain-string: explanation in english prose
    answers["(a) example"] = "To guarantee that the rules do not overlap, it's important that the conditions they set forth are distinct from one another. This appears to be the case, as they classify different categories—like birds, fish, mammals, and reptiles—each defined by its own specific set of characteristics."
    answers["(b) example"] = 'The rules do not include a category for amphibians, meaning that according to the current guidelines, a salamander would not be categorized. This suggests that the rules are incomplete, failing to account for all vertebrates in the dataset.'
    answers["(c) example"] = 'Since these rules are not mutually exclusive, the order of the rules matters. For example, if the first rule is true, the second rule will not be evaluated. If the first rule is false, the second rule will be evaluated. Therefore, the order of the rules matters.'

    return answers
# -----------------------------------------------------------
def question7():
    answers = {}

    # bool: True/False
    answers["(a)"] = False
    answers["(b)"] = True
    answers["(c)"] = False
    answers["(d)"] = True

    # explain_string: explanation in english prose
    answers["(a) explain"] = 'In the backward phase, the weight update formula is applied in the reverse direction. In other words, the weights at level k + 1 are updated before the weights at level k are updated'
    answers["(b) explain"] = 'Because the activation results of the kth layer become the input to the k+1 layer, and sicne these inputs are then multiplied to weights and added together for eacg neuron and agin we calculate activation for each neuron in the k+1 layer'
    answers["(c) explain"] = 'The vanishing gradient problem occurs when gradients become very small during backpropagation through many neural network layers.'
    answers["(d) explain"] = 'Yes because the gradient of loss is only 0 are minima or maxima, so if the gradient of loss will be zero for training data.'

    return answers

# -----------------------------------------------------------
def question8():
    answers = {}

    # float
    answers["(a) P(X_1=1)"] = 0.65
    answers["(a) P(X_2=1)"] = 0.41
    answers["(a) P(X_1=1,X_2=1)"] = 0.28

    # string: "dependent" or "independent"
    answers["(a) Relationship between X_1 and X_2"] = 'dependent'

    # string: "yes" or "no"
    answers["(b) X_1 and X_2 conditionally independent given the class?"] = 'yes'

    # float
    answers["(c) P(X_1=1 | +)"] = 0.8
    answers["(c) P(X_1=1 | -)"] = 0.5
    answers["(c) P(X_2=1 | +)"] = 0.5
    answers["(c) P(X_2=1 | -)"] = 0.32
    answers["(c) P(X_3=1 | +)"] = 0.4
    answers["(c) P(X_3=1 | -)"] = 0.16

    # For each row give the class predicted by the model after training using Naive Bayes
    # String: either '+' or '-'
    answers["(d) Row 1"] = '+'
    answers["(d) Row 2"] = '-'
    answers["(d) Row 3"] = '-'
    answers["(d) Row 4"] = '-'

    # float between 0 and 1
    answers["(d) Training error rate"] = 0.20

    return answers

# --------------------------------------------------------
def question9():
    answers = {}

    # int
    answers["(a) K"] = 5
    answers["(b) K"] = 5

    # explain_string
    answers["(a) explain"] = 'The instances are well separated into two distinct clusters, so the best value of K should be small but 1 would be too sensitive to noise. therefore k=5 is a good choice.'
    answers["(b) explain"] = 'due to the overlap, a slightly larger K than 5, but not as large as 50, might be most effective, so k=5 or slightly more is a good choice.'

    return answers

# --------------------------------------------------------
def question10():
    answers = {}

    # float
    answers["(a) P(A=1|+)"] = 0.6
    answers["(a) P(B=1|+)"] = 0.4
    answers["(a) P(C=1|+)"] = 0.8
    answers["(a) P(A=1|-)"] = 0.4
    answers["(a) P(B=1|-)"] = 0.4
    answers["(a) P(C=1|-)"] = 0.2

    # type: explanatory string
    answers["(a) P(A=1|+) explain your answer"] = 'the probability of A=1 given class + is the number of instances where A=1 and class + divided by the total number of instances where class +, so i just counted the number of instances where A=1 and class + and divided by the total number of instances where class +. Similarly is done for the other probabilities as well'
  
    # type: float
    # note: R is the sample (A=1,B=1,C=1)
    answers["(b) P(+|R)"] = 0.96 
    answers["(b) P(R|+)"] = 0.192
    answers["(b) P(R|-)"] = 0.032

    # string, '+' or '-'
    answers["(b) class label"] = '+'

    # explain_string
    answers["(b) Explain your reasoning"] = 'Naive Bayes of P(A=1|+)* P(B=1|+)* P(C=1|+) is 0.192 whereas for P(A=1|-)* P(B=1|-)* P(C=1|-) it is 0.032'
  
    # float
    answers["(c) P(A=1)"] = 0.5
    answers["(c) P(B=1)"] = 0.4
    answers["(c) P(A=1,B=1)"] = 0.2

    # type: string, 'yes' or 'no'
    answers["(c) A independent of B?"] = 'yes'
  
    # type: float
    answers["(d) P(A=1)"] = 0.5
    answers["(d) P(B=0)"] = 0.6
    answers["(d) P(A=1,B=0)"] = 0.3

    # type: string: 'yes' or 'no'
    answers["(d) A independent of B?"] = 'yes'
  
    # type: float
    answers["(e) P(A=1,B=1|+)"] = 0.2
    answers["(e) P(A=1|+)"] = 0.6
    answers["(e) P(B=1|+)"] = 0.4

    # type: string: 'yes' or 'no'
    answers["(e) A independent of B given class +?"] = 'no'

    # type: explanatory string
    answers["(e) A and B conditionally independent given class +, explain"] =  'no, because the probability of A=1 and B=1 given class + is not equal to the product of the probabilities of A=1 and B=1 given class +, so they are not conditionally independent given class +.'
  
    return answers
# --------------------------------------------------------
if __name__ == '__main__':
    answers_dict = {}
    answers_dict['question1'] = question1()
    # answers_dict['question2'] = question2()
    answers_dict['question3'] = question3()
    # answers_dict['question4'] = question4()
    answers_dict['question7'] = question7()
    answers_dict['question8'] = question8()
    answers_dict['question9'] = question9()
    answers_dict['question10'] = question10()

    with open('answers.pkl', 'wb') as f:
        pickle.dump(answers_dict, f)
