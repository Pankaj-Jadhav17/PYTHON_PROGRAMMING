#!/usr/bin/env python3
import math

def gini(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    return 1.0 - sum((c/total)**2 for c in counts)

def entropy(counts, base=10):
    total = sum(counts)
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c == 0:
            continue
        p = c/total
        ent -= p * math.log(p, base)
    return ent

def weighted_metric(left_counts, right_counts, metric_fn, base=None):
    left_total = sum(left_counts)
    right_total = sum(right_counts)
    total = left_total + right_total
    left_val = metric_fn(left_counts) if base is None else metric_fn(left_counts, base)
    right_val = metric_fn(right_counts) if base is None else metric_fn(right_counts, base)
    return (left_total/total)*left_val + (right_total/total)*right_val, left_val, right_val

def print_rule(name, left_counts, right_counts):
    print(f"--- {name} ---")
    g_weighted, g_left, g_right = weighted_metric(left_counts, right_counts, gini)
    e_weighted_10, e_left_10, e_right_10 = weighted_metric(left_counts, right_counts, entropy, base=10)
    e_parent_10 = entropy([left_counts[0]+right_counts[0], left_counts[1]+right_counts[1]], base=10)
    info_gain_10 = e_parent_10 - e_weighted_10
    print(f"Gini left: {g_left:.4f}, right: {g_right:.4f}, weighted Gini: {g_weighted:.4f}")
    print(f"Entropy (base10) parent: {e_parent_10:.4f}, left: {e_left_10:.4f}, right: {e_right_10:.4f}, weighted entropy: {e_weighted_10:.4f}, info gain: {info_gain_10:.4f}")
    print()

def main():
    # Rule1: if D1<=5 left:45 (c1:35,c2:10) else right:55 (c1:15,c2:40)
    rule1_left = [35,10]
    rule1_right = [15,40]
    print_rule("Rule1 (D1<=5)", rule1_left, rule1_right)
    # Rule2: if D2<=3 left:40 (30,10) else right:60 (20,40)
    rule2_left = [30,10]
    rule2_right = [20,40]
    print_rule("Rule2 (D2<=3)", rule2_left, rule2_right)
    # Rule3: if D3<=7 left:50 (45,5) else right:50 (5,45)
    rule3_left = [45,5]
    rule3_right = [5,45]
    print_rule("Rule3 (D3<=7)", rule3_left, rule3_right)

if __name__ == "__main__":
    main()







# User: total eg : 100
# c1:50
# c2:50

# ####### 1st rule #############

# if D1<= 5
#     45 -> c1:35, c2:10
# else (D1 > 5)
#     55 -> c1:15, c2:40

# # Gini
# Gini_split1 = 1 - (35/45)^2 - (10/45)^2 = 0.3456
# Gini_split2 = 1 - (15/55)^2 - (40/55)^2 = 0.3966

# weighted Gini_index = (45/100) * Gini_split1 + (55/100) * Gini_split2
# # = 0.3721          = (45/100) * 0.3456 + (55/100) * 0.3966 = 0.3736

# # Entropy

# #Entropy of parent node
# Entropy_parent = - (50/100) * log(50/100) - (50/100) * log(50/100) = 0.301

# # Entropy of splits / children nodes
# Entropy_split1 = - (35/45) * log2(35/45) - (10/45) * log2(10/45) = 0.2300
# Entropy_split2 = - (15/55) * log2(15/55) - (40/55) * log2(40/55) = 0.2545

# weighted Entropy_of child = (45/100) * Entropy_split1 + (55/100) * Entropy_split2
# # = 0.2435          = (45/100) * 0.2300 + (55/100) * 0.2545 = 0.2435    

# information Gain = Entropy_parent - weighted Entropy_of child
# # = 0.0575          = 0.301 - 0.2435

# ######## 2nd rule #############
# if D2<= 3
#     40 -> c1:30, c2:10
# else (D2 > 3)
#     60 -> c1:20, c2:40
    
# # Gini
# Gini_split1 = 1 - (30/40)^2 - (10/40)^2 = 
# Gini_split2 = 1 - (20/60)^2 - (40/60)^2 =   

# weighted Gini_index = (40/100) * Gini_split1 + (60/100) * Gini_split2

# # Entropy
# Entropy_split1 = - (30/40) * log2(30/40) - (10/40) * log2(10/40) =
# Entropy_split2 = - (20/60) * log2(20/60) - (40/60) * log2(40/60) =

# weighted Entropy = (40/100) * Entropy_split1 + (60/100) * Entropy_split2

# Information Gain = 1- weighted Entropy


# ####### 3rd rule #############

# if D3<= 7
#     50 -> c1:45, c2:5
# else (D3 > 7)   
#     50 -> c1:5, c2:45
    
# # Gini
# Gini_split1 = 1 - (45/50)^2 - (5/50)^2 =
# Gini_split2 = 1 - (5/50)^2 - (45/50)^2 =

# weighted Gini_index = (50/100) * Gini_split1 + (50/100) * Gini_split2

# # Entropy
# Entropy_split1 = - (45/50) * log2(45/50) - (5/50) * log2(5/50) =
# Entropy_split2 = - (5/50) * log2(5/50) - (45/50) * log2(45/50) =

# weighted Entropy = (50/100) * Entropy_split1 + (50/100) * Entropy_split2

# this code gives an syntax error

# GitHub Copilot: Optimizing tool selection...I'll open DT_gini_entropy.py to inspect the code and find the syntax error.

# Read [](file:///home/pankaj/Desktop/PYTHON_PROGRAMMING/PYTHON_PROGRAMMING/ML/DT_gini_entropy.py)

# I'll create a short TODO plan: replace the invalid pseudo-code with a runnable Python script that computes Gini and Entropy for the three rules.

# Created 3 todos

# Now I'll replace the pseudo-code in DT_gini_entropy.py with a runnable Python script that computes Gini and entropy (base-10 to match your numbers), then run it.

# Using "Apply Patch"


# ```

# ```


# Ran terminal command:  python3 DT_gini_entropy.py

# Made changes.