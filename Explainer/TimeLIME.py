import itertools
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
from tqdm import tqdm
import re

def TimeLIME(train, test, model, output_path, top=5):
    """
    Based on https://github.com/ai-se/TimeLIME and https://github.com/kpeng2019/TimeLIME
    modified to work with our dataset
    - no normalization
    - no (k+1) release information when calculating historical feature changes

    """
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
        if i in deltas[0:top]:
            actionable.append(1)
        else:
            actionable.append(0)

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
        changes = [(idx, change) for idx, change in enumerate(changes) if change != 0]
        changes = [
            "inc" + str(item[0]) if item[1] > 0 else "dec" + str(item[0])
            for item in changes
        ]
        if len(changes) > 0:
            itemsets.append(changes)

    te = TransactionEncoder()
    te_ary = te.fit(itemsets).transform(itemsets, sparse=True)
    df = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)
    rules = apriori(df, min_support=0.001, max_len=top, use_colnames=True)

    min_val = X_train.min()
    max_val = X_train.max()

    predictions = model.predict(X_test.values)

    for i in tqdm(
        range(0, len(y_test)),
        desc="Generating explanations",
        leave=False,
        total=len(y_test),
    ):
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

        plan, rec = flip(
            temp,
            ins.as_list(label=1),
            ind,
            X_train.columns,
            actionable,
            min_val,
            max_val,
        )

        if rec in seen_id:
            supported_plan_id = seen[seen_id.index(rec)]
        else:
            supported_plan_id = find_supported_plan(rec, rules, top=top)
            seen_id.append(rec.copy())

            seen.append(supported_plan_id)

        supported_plan = []
        for k in range(len(rec)):
            if rec[k] != 0:
                if (k not in supported_plan_id) and ((0 - k) not in supported_plan_id):
                    rec[k] = 0

            if rec[k] != 0:
                feature_name = X_train.columns[k]
                importance = [
                    (i, pair[1]) for i, pair in enumerate(ind) if pair[0] == k
                ][0]
                idx = importance[0]
                importance = importance[1]
                interval = ins.as_list(label=1)[idx][0]
                supported_plan.append(
                    [
                        feature_name,
                        temp[k],
                        importance,
                        plan[k][0],
                        plan[k][1],
                        rec[k],
                        interval,
                        min_val[k],
                        max_val[k],
                    ]
                )
        supported_plan = sorted(supported_plan, key=lambda x: abs(x[2]), reverse=True)
        result_df = pd.DataFrame(
            supported_plan,
            columns=[
                "feature",
                "value",
                "importance",
                "left",
                "right",
                "rec",
                "rule",
                "min",
                "max",
            ],
        )
        result_df.to_csv(file_name, index=False)


# Modify the flip function to handle non-normalized features while considering their min-max values
def flip(data_row, local_exp, ind, cols, actionable, min_val, max_val):
    cache = []
    trans = []
    cnt, cntp, cntn = [], [], []

    for i in range(0, len(local_exp)):
        cache.append(ind[i])
        trans.append(local_exp[i])

        if ind[i][1] > 0:
            cntp.append(i)
            cnt.append(i)
        else:
            cntn.append(i)
            cnt.append(i)

    record = [0 for n in range(len(cols))]
    tem = data_row.copy()
    result = [[0 for m in range(2)] for n in range(len(cols))]

    pattern = re.compile(
        r"([-]?[\d.]+)?\s*(<|>)?\s*([a-zA-Z_]+)\s*(<=|>=|<|>)?\s*([-]?[\d.]+)?"
    )

    for j in range(0, len(local_exp)):
        act = True
        index = cache[j][0]
        if actionable:
            if actionable[index] == 0:
                act = False

        match pattern.search(trans[j][0]).groups():
            case v1, "<", feature_name, "<=", v2:
                l, r = float(v1), float(v2)
            case None, None, feature_name, ">", v1:
                l, r = float(v1), max_val[cache[j][0]]
            case None, None, feature_name, "<=", v2:
                l, r = min_val[cache[j][0]], float(v2)
        assert feature_name == cols[cache[j][0]]

        if j in cnt and act:
            if j in cntp:
                result[index][0], result[index][1] = min_val[index], l
                record[index] = -1
            else:
                result[index][0], result[index][1] = r, max_val[index]
                record[index] = 1
        else:
            result[index][0], result[index][1] = l, r

    return result, record


def hedge(arr1, arr2):
    # returns a value, larger means more changes
    s1, s2 = np.std(arr1), np.std(arr2)
    m1, m2 = np.mean(arr1), np.mean(arr2)
    n1, n2 = len(arr1), len(arr2)
    num = (n1 - 1) * s1**2 + (n2 - 1) * s2**2
    denom = n1 + n2 - 1 - 1
    sp = (num / denom) ** 0.5
    delta = np.abs(m1 - m2) / sp
    c = 1 - 3 / (4 * (denom) - 1)
    return delta * c


def get_support(string, rules):
    for i in range(rules.shape[0]):
        if set(rules.iloc[i, 1]) == set(string):
            return rules.iloc[i, 0]
    return 0


def find_supported_plan(plan, rules, top=5):
    proposed = []
    max_change = top
    max_sup = 0
    result_id = []
    pool = []
    for j in range(len(plan)):
        if plan[j] == 1:
            result_id.append(j)
            proposed.append("inc" + str(j))
        elif plan[j] == -1:
            result_id.append(-j)
            proposed.append("dec" + str(j))
    while max_sup == 0:
        pool = list(itertools.combinations(result_id, max_change))
        for each in pool:
            temp = []
            for k in range(len(each)):
                if each[k] > 0:
                    temp.append("inc" + str(each[k]))
                elif each[k] < 0:
                    temp.append("dec" + str(-each[k]))
            temp_sup = get_support(temp, rules)
            if temp_sup > max_sup:
                max_sup = temp_sup
                result_id = each
        max_change -= 1
        if max_change <= 0:
            print("Failed!!!")
            break
    return result_id
