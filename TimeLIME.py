import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import time
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import MinMaxScaler
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
from tqdm import tqdm

def translate(rule, name, dtype):
    items = rule.split(name)
    items = [item.strip() for item in items]
    for item in items:
        item = item.strip("<")
        item = item.strip("<=")
        item = item.strip(">")
        item = item.strip(">=")
        item = item.strip()
    return items



def translate1(sentence,name):
    # do not aim to change the column
    lst = sentence.strip().split(name)
    left,right= 0,0
    if lst[0] == '':
        del lst[0]
    if len(lst)==2:
        if '<=' in lst[1]:
            aa=lst[1].strip(' <=')
            right= float(aa)
        elif '<' in lst[1]:
            aa=lst[1].strip(' <')
            right = float(aa)
        if '<=' in lst[0]:
            aa=lst[0].strip(' <=')
            left = float(aa)
        elif '<' in lst[0]:
            aa=lst[0].strip(' <')
            left = float(aa)
    else:
        if '<=' in lst[0]:
            aa=lst[0].strip(' <=')
            right = float(aa)
            left = 0
        elif '<' in lst[0]:
            aa=lst[0].strip(' <')
            right = float(aa)
            left = 0
        if '>=' in lst[0]:
            aa=lst[0].strip(' >=')
            left = float(aa)
            right = 1
        elif '>' in lst[0]:
            aa=lst[0].strip(' >')
            left = float(aa)
            right = 1
    return left, right

def TL(train, test, model):
    start_time = time.time()

    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]
    X_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1]    

    deltas = []
    for col in X_train.columns:
        deltas.append(hedge(X_train[col].values, X_test[col].values))
    deltas = sorted(range(len(deltas)), key=lambda k: deltas[k], reverse=True)

    actionable = []
    for i in range(0, len(deltas)):
        if i in deltas[0:5]:
            actionable.append(1)
        else:
            actionable.append(0)
    # print(actionable)

    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        training_labels=y_train.values,
        feature_names=X_train.columns,
        discretizer="entropy",
        feature_selection="lasso_path",
        mode="classification",
    )

    records = []
    seen = []
    seen_id = []    
    plans = []
    instances = []
    importances = []
    indices = []

    itemsets = []
    common_indices = X_test.index.intersection(X_train.index)
    for name in tqdm(common_indices, desc="Generating itemsets", leave=False, total=len(common_indices)):
    
        changes = X_test.loc[name].values - X_train.loc[name].values
        changes = [
            (idx, change) for idx, change in enumerate(changes) if change != 0
        ]
        changes = [
            "inc" + str(item[0]) if item[1] > 0 else "dec" + str(item[0])
            for item in changes
        ]
        if len(changes) > 0:
            itemsets.append(changes)
    print("Number of itemsets:", len(itemsets))
    te = TransactionEncoder()
    te_ary = te.fit(itemsets).transform(itemsets, sparse=True)
    df = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)
    rules = apriori(df, min_support=0.001, max_len=5, use_colnames=True)

    predictions = model.predict(X_test.values)
    for i in range(0, len(y_test)):
        real_target = predictions[i]
        if real_target == 1 and y_test.values[i] == 1:
            ins = explainer.explain_instance(
                data_row=X_test.values[i],
                predict_fn=model.predict_proba,
                num_features=len(X_train.columns),
                num_samples=5000,
            )
            ind = ins.local_exp[1]
            temp = X_test.values[i].copy()

            plan, rec = flip(
                temp, ins.as_list(label=1), ind, X_train.columns, actionable
            )

            if rec in seen_id:
                supported_plan_id = seen[seen_id.index(rec)]
            else:
                supported_plan_id = find_supported_plan(rec, rules, top=5)
                seen_id.append(rec.copy())

                seen.append(supported_plan_id)

            
            for k in range(len(rec)):
                if rec[k] != 0:
                    if (k not in supported_plan_id) and (
                        (0 - k) not in supported_plan_id
                    ):
                        plan[k][0], plan[k][1] = temp[k] - 0.05, temp[k] + 0.05
                        rec[k] = 0

            records.append(rec)
            plans.append(plan)
            instances.append(temp)
            importances.append(ind)
            indices.append(X_test.index[i])
            
    print("Runtime:", time.time() - start_time)

    return records, plans, instances, importances, indices

def get_feature_index(feature_names, feature):
    for i in range(0, len(feature_names)):
        if feature_names[i] == feature:
            return i
    return -1

def flip(data_row,local_exp,ind, cols, actionable):
    cache = []
    trans = []
    # Store feature index in cache.
    cnt, cntp,cntn = [],[],[]
    for i in range(0,len(local_exp)):
        cache.append(ind[i])
        trans.append(local_exp[i])
        
        if ind[i][1]>0:
            cntp.append(i)
            cnt.append(i)
        elif ind[i][1]<0:
            cntn.append(i)
            cnt.append(i)
    
    record =[0 for n in range(len(cols))]
    tem = data_row.copy()
    result =  [[ 0 for m in range(2)] for n in range(len(cols))]
    for j in range(0,len(local_exp)):
        act = True
        index = cache[j][0]
        if actionable:
            if actionable[index]==0:
                act=False
        if j in cnt and act:
            if j in cntp:
                result[index][0],result[index][1] = 0,tem[index]
                record[index]=-1
            else:
                result[index][0],result[index][1] = tem[index],1
                record[index]=1
        else:
            if act:
                result[index][0],result[index][1] = tem[index]-0.005,tem[index]+0.005
            else:
                result[index][0],result[index][1] = tem[index]-0.05,tem[index]+0.05

    return result,record


def TimeLIME(train, test, model, output_path):
    start_time = time.time()

    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]
    X_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1]    

    deltas = []
    for col in X_train.columns:
        deltas.append(hedge(X_train[col].values, X_test[col].values))
    deltas = sorted(range(len(deltas)), key=lambda k: deltas[k], reverse=True)

    actionable = []
    for i in range(0, len(deltas)):
        if i in deltas[0:5]:
            actionable.append(1)
        else:
            actionable.append(0)
    # print(actionable)

    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        training_labels=y_train.values,
        feature_names=X_train.columns,
        discretizer="entropy",
        feature_selection="lasso_path",
        mode="classification",
    )


    seen = []
    seen_id = []    

    itemsets = []
    common_indices = X_test.index.intersection(X_train.index)
    for name in common_indices:
    
        changes = X_test.loc[name].values - X_train.loc[name].values
        changes = [
            (idx, change) for idx, change in enumerate(changes) if change != 0
        ]
        changes = [
            "inc" + str(item[0]) if item[1] > 0 else "dec" + str(item[0])
            for item in changes
        ]
        if len(changes) > 0:
            itemsets.append(changes)

    te = TransactionEncoder()
    te_ary = te.fit(itemsets).transform(itemsets, sparse=True)
    df = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)
    rules = apriori(df, min_support=0.001, max_len=5, use_colnames=True)

    min_val = X_train.min()
    max_val = X_train.max()

    predictions = model.predict(X_test.values)
    for i in tqdm(range(0, len(y_test)), desc="Generating explanations", leave=False, total=len(y_test)):
        file_name = output_path / f"{X_test.index[i]}.csv"
        if file_name.exists():
            continue
        real_target = predictions[i]
        if real_target == 0 or y_test.values[i] == 0:
            continue

        ins = explainer.explain_instance(
            data_row=X_test.values[i],
            predict_fn=model.predict_proba,
            num_features=len(X_train.columns),
            num_samples=5000,
        )
        ind = ins.local_exp[1]
        temp = X_test.values[i].copy()
        rule_pairs = ins.as_list(label=1)

        plan, rec = flip_non_normalized(
            temp, ins.as_list(label=1), ind, X_train.columns, actionable, min_val, max_val
        )

        if rec in seen_id:
            supported_plan_id = seen[seen_id.index(rec)]
        else:
            supported_plan_id = find_supported_plan(rec, rules, top=5)
            seen_id.append(rec.copy())

            seen.append(supported_plan_id)

        supported_plan = []
        for k in range(len(rec)):
            if rec[k] != 0:
                if (k not in supported_plan_id) and (
                    (0 - k) not in supported_plan_id
                ):
                    perturbation_range = 0.05 * (max_val[k] - min_val[k])
                    plan[k][0], plan[k][1] = temp[k] - perturbation_range, temp[k] + perturbation_range
                    rec[k] = 0
            
            if rec[k] != 0:
                feature_name = X_train.columns[k]
                importance = [ pair[1] for pair in ind if pair[0] == k][0]
                interval = [ pair[0] for pair in ins.as_list(label=1) if feature_name in pair[0]][0]
                supported_plan.append([
                    X_train.columns[k],
                    temp[k],
                    importance,
                    plan[k][0],
                    plan[k][1],
                    rec[k],
                    interval
                ])
        supported_plan = sorted(supported_plan, key=lambda x: abs(x[2]), reverse=True)
        result_df = pd.DataFrame(supported_plan, columns=["feature", "value", "importance", "left", "right", "rec", "rule"])
        result_df.to_csv(file_name, index=False)

  

# Modify the flip function to handle non-normalized features while considering their min-max values
def flip_non_normalized(data_row, local_exp, ind, cols, actionable, min_val, max_val):
    cache = []
    trans = []
    cnt, cntp, cntn = [], [], []
    
    for i in range(0, len(local_exp)):
        cache.append(ind[i])
        trans.append(local_exp[i])
        
        if ind[i][1] > 0:
            cntp.append(i)
            cnt.append(i)
        elif ind[i][1] < 0:
            cntn.append(i)
            cnt.append(i)
    
    record = [0 for n in range(len(cols))]
    tem = data_row.copy()
    result = [[0 for m in range(2)] for n in range(len(cols))]
    
    for j in range(0, len(local_exp)):
        act = True
        index = cache[j][0]
        if actionable:
            if actionable[index] == 0:
                act = False
        
        if j in cnt and act:
            if j in cntp:
                result[index][0],result[index][1] = min_val[index],tem[index]
                record[index]=-1
            else:
                result[index][0],result[index][1] = tem[index],max_val[index]
                record[index]=1
        else:
            if act:
                # Just a small perturbation around the original value for non-actionable features
                perturbation_range = 0.005 * (max_val[index] - min_val[index])
                result[index][0], result[index][1] = tem[index] - perturbation_range, tem[index] + perturbation_range
            else:
                # A larger perturbation for non-actionable features
                perturbation_range = 0.05 * (max_val[index] - min_val[index])
                result[index][0], result[index][1] = tem[index] - perturbation_range, tem[index] + perturbation_range

    return result, record


def flip_interval(l, r, min_val, max_val, dtype):
    # Normalize the interval to [0, 1]
    normalized_l = (l - min_val) / (max_val - min_val)
    normalized_r = (r - min_val) / (max_val - min_val)

    # Flip the interval
    flipped_n_l, flipped_n_r = 1 - normalized_r, 1 - normalized_l

    # Denormalize the interval
    flipped_l = flipped_n_l * (max_val - min_val) + min_val
    flipped_r = flipped_n_r * (max_val - min_val) + min_val

    if dtype == int:
        flipped_l = int(round(flipped_l))
        flipped_r = int(round(flipped_r))

    return flipped_l, flipped_r

def hedge(arr1,arr2):
    # returns a value, larger means more changes
    s1,s2 = np.std(arr1),np.std(arr2)
    m1,m2 = np.mean(arr1),np.mean(arr2)
    n1,n2 = len(arr1),len(arr2)
    num = (n1-1)*s1**2 + (n2-1)*s2**2
    denom = n1+n2-1-1
    sp = (num/denom)**.5
    delta = np.abs(m1-m2)/sp
    c = 1-3/(4*(denom)-1)
    return delta*c


def get_support(string, rules):
    # print(string)
    for i in range(rules.shape[0]):
        if set(rules.iloc[i,1]) == set(string):
            return rules.iloc[i,0]
    return 0


def find_supported_plan(plan,rules,top=5):
    proposed = []
    max_change=top
    max_sup=0
    result_id=[]
    pool=[]
    for j in range(len(plan)):
        if plan[j]==1:
            result_id.append(j)
            proposed.append("inc"+str(j))
        elif plan[j]==-1:
            result_id.append(-j)
            proposed.append("dec"+str(j))
#     if max_change==top:
#         max_sup = get_support(proposed,rules)
    while (max_sup==0):
        pool = list(itertools.combinations(result_id, max_change))
        for each in pool:
            temp = []
            for k in range(len(each)):
                if each[k]>0:
                    temp.append("inc"+str(each[k]))
                elif each[k]<0:
                    temp.append("dec"+str(-each[k]))
#             print('temp',temp)
            temp_sup = get_support(temp,rules)
            if temp_sup>max_sup:
                max_sup = temp_sup
                result_id = each
        max_change-=1
        if max_change<=0:
            print("Failed!!!")
            break
    return result_id