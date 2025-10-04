import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb
import re
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# =========================
# 1. 读取数据
# =========================
path = "C:\\Users\\86178\\Desktop\\math\\"
df = pd.read_excel(path + '附件.xlsx', sheet_name='女胎检测数据')

def week_to_days(week_str):
    """
    支持 '11w+6', '20w+1', '23w' 等格式
    转换为天数：周数*7 + 天数
    """
    s = str(week_str).strip().lower()
    # 先匹配带 + 天数的
    match = re.match(r"(\d+)\s*w\s*\+\s*(\d+)", s)
    if match:
        weeks, days = int(match.group(1)), int(match.group(2))
        return weeks * 7 + days
    # 再匹配只有周数的
    match = re.match(r"(\d+)\s*w", s)
    if match:
        weeks = int(match.group(1))
        return weeks * 7
    return None


df["检测孕周"] = df["检测孕周"].apply(week_to_days)
df['异常情况'] = df['染色体的非整倍体'].apply(lambda x: 0 if pd.isna(x) else 1)

col = '染色体的非整倍体'

# 创建新列
df['T13异常'] = df[col].apply(lambda x: 1 if pd.notna(x) and 'T13' in x else 0)
df['T18异常'] = df[col].apply(lambda x: 1 if pd.notna(x) and 'T18' in x else 0)
df['T21异常'] = df[col].apply(lambda x: 1 if pd.notna(x) and 'T21' in x else 0)
df['基因段有效率'] = df['在参考基因组上比对的比例'] * (1 - df['重复读段的比例']) * (1 - df['被过滤掉读段数的比例'])



# Z值取绝对值（不归一化）

z_cols = ['X染色体的Z值', '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值']
for col in z_cols:
    if col in df.columns:
        df[col] = df[col].abs()


# 全局归一化（不包括 Z 值列）

scale_cols = [
    '孕妇BMI', '检测孕周', '原始读段数',
    '唯一比对的读段数'
]

scaler = MinMaxScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])


# 异常列对应保留 columns
abnormal_columns_map = {
    'T13异常': [
        'X染色体的Z值', '孕妇BMI', '检测孕周', 'GC含量', 'X染色体的Z值', 
        '13号染色体的Z值', '13号染色体的GC含量','原始读段数', '重复读段的比例', '被过滤掉读段数的比例'
    ],
    'T18异常': [
        '孕妇BMI', '检测孕周', 'GC含量', 'X染色体的Z值', 
        '唯一比对的读段数', '18号染色体的Z值', '18号染色体的GC含量','在参考基因组上比对的比例',
        '原始读段数', '重复读段的比例', '被过滤掉读段数的比例'
    ],
    'T21异常': [
        '孕妇BMI', '检测孕周', 'GC含量', 'X染色体的Z值', 
        '唯一比对的读段数', '21号染色体的Z值', '21号染色体的GC含量','在参考基因组上比对的比例',
        '原始读段数', '重复读段的比例', '被过滤掉读段数的比例'
    ]
}


#  构建序列数据
def build_sequences(df, abnormal_col, max_len=4):
    cols = abnormal_columns_map[abnormal_col]
    patient_ids = df['孕妇代码'].unique()
    
    X, y = [], []
    for pid in patient_ids:
        sub = df[df['孕妇代码'] == pid].sort_values('检测日期')
        seq = sub[cols].to_numpy()
        
        # padding / 截断
        if len(seq) < max_len:
            pad = np.full((max_len - len(seq), seq.shape[1]), np.nan)
            seq = np.vstack([seq, pad])
        else:
            seq = seq[:max_len]
        
        # 用 DataFrame 填充 NaN
        seq = pd.DataFrame(seq, columns=cols).ffill().bfill().to_numpy()
        
        X.append(seq)
        y.append(sub[abnormal_col].max())
    
    return np.array(X), np.array(y)


# 构建 T13/T18/T21 数据集

X_T13, y_T13 = build_sequences(df, 'T13异常')
X_T18, y_T18 = build_sequences(df, 'T18异常')
X_T21, y_T21 = build_sequences(df, 'T21异常')

print("T13:", X_T13.shape, y_T13.shape)
print("T18:", X_T18.shape, y_T18.shape)
print("T21:", X_T21.shape, y_T21.shape)

# 检查是否还有 NaN
print("NaN 检查:", np.isnan(X_T13).sum(), np.isnan(X_T18).sum(), np.isnan(X_T21).sum())

def undersample_neg_three_times_pos(X, y, random_state=42):
    """
    将多数类（负样本）下采样到正样本数量的 3 倍
    X: np.array, shape=(n_samples, seq_len, n_features)
    y: np.array, shape=(n_samples,)
    返回下采样后的 X_res, y_res
    """
    np.random.seed(random_state)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    
    if n_pos == 0 or n_neg == 0:
        return X, y  # 无法下采样
    
    # 负样本下采样到正样本数量的 3 倍
    n_neg_target = n_pos * 6
    if n_neg > n_neg_target:
        neg_idx_down = np.random.choice(neg_idx, n_neg_target, replace=False)
    else:
        neg_idx_down = neg_idx  # 少于目标，不处理
    
    keep_idx = np.concatenate([pos_idx, neg_idx_down])
    np.random.shuffle(keep_idx)
    
    return X[keep_idx], y[keep_idx]

X_T13_res, y_T13_res = undersample_neg_three_times_pos(X_T13, y_T13)
X_T18_res, y_T18_res = undersample_neg_three_times_pos(X_T18, y_T18)
X_T21_res, y_T21_res = undersample_neg_three_times_pos(X_T21, y_T21)

print("T13下采样后:", X_T13_res.shape, y_T13_res.shape)
print("T18下采样后:", X_T18_res.shape, y_T18_res.shape)
print("T21下采样后:", X_T21_res.shape, y_T21_res.shape)

datasets = {
    'T13': (X_T13, y_T13),
    'T18': (X_T18, y_T18),
    'T21': (X_T21, y_T21)
}

for name, (X, y) in datasets.items():
    print(f"\n==================== {name} ====================")
    
    # 划分训练集/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 定义模型
    model = Sequential([
        Masking(mask_value=0.0, input_shape=(X.shape[1], X.shape[2])),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # 训练
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=16,
        verbose=1
    )
    
    # 预测与评估
    y_pred_prob = model.predict(X_test).ravel()
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_pred_prob))

    
    # 画 ROC 曲线
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{name} ROC Curve')
    plt.legend(loc='lower right')
    plt.show()


# 每个异常对应保留的 columns
abnormal_columns_map = {
    'T13异常': ['孕妇BMI','GC含量','X染色体的Z值','唯一比对的读段数','基因段有效率',
                '13号染色体的Z值','13号染色体的GC含量'],
    'T18异常': ['孕妇BMI','GC含量','X染色体的Z值','唯一比对的读段数','基因段有效率',
                '18号染色体的Z值','18号染色体的GC含量'],
    'T21异常': ['孕妇BMI','GC含量','X染色体的Z值','唯一比对的读段数','基因段有效率',
                '21号染色体的Z值','21号染色体的GC含量']
}

def process_abnormal(df, abnormal_col):
    cols = abnormal_columns_map[abnormal_col]

    def compute_stats(group):
        abnormal_rows = group[group[abnormal_col] == 1]
        if len(abnormal_rows) == 0:
            data = group[cols]  # 没异常，用所有数据
            abnormal_flag = 0
        elif len(abnormal_rows) == 1:
            data = pd.DataFrame([abnormal_rows.iloc[0][cols]])  # 单条异常
            abnormal_flag = 1
        else:
            data = abnormal_rows[cols]  # 多条异常
            abnormal_flag = 1

        # 计算每列的 mean/max/min
        stats = {}
        for col in cols:
            stats[f"{abnormal_col}_{col}_mean"] = data[col].mean()
            stats[f"{abnormal_col}_{col}_max"] = data[col].max()
            stats[f"{abnormal_col}_{col}_min"] = data[col].min()

        stats['孕妇代码'] = group['孕妇代码'].iloc[0]  # 添加孕妇代码
        stats[abnormal_col] = abnormal_flag             # 添加异常标记
        return pd.Series(stats)

    return df.groupby('孕妇代码').apply(compute_stats).reset_index(drop=True)

# 生成三个异常的特征表
T13_df = process_abnormal(df, 'T13异常')
T18_df = process_abnormal(df, 'T18异常')
T21_df = process_abnormal(df, 'T21异常')

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score

def train_xgb(df, target_col):
    # 分离特征和标签
    X = df.drop(columns=['孕妇代码', 'T13异常', 'T18异常', 'T21异常'], errors='ignore')
    y = df[target_col]

    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 处理正负样本不平衡：scale_pos_weight = neg/pos
    n_pos = sum(y_train==1)
    n_neg = sum(y_train==0)
    scale_pos_weight = n_neg / n_pos

    # 定义基础模型
    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        use_label_encoder=False,
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )

    # 网格搜索参数
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 1],
        'colsample_bytree': [0.8, 1]
    }

    grid_search = GridSearchCV(
        xgb_clf, param_grid, scoring='roc_auc', cv=3, n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)

    # 最优模型
    best_model = grid_search.best_estimator_

    # 测试集预测
    y_pred_prob = best_model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_pred_prob)

    print(f"{target_col} 最优参数: {grid_search.best_params_}")
    print(f"{target_col} 测试集 AUC: {auc:.4f}")

    return best_model, auc, grid_search.best_params_

# 假设 T13_df/T18_df/T21_df 已生成，并包含对应异常列
T13_model, T13_auc, T13_params = train_xgb(T13_df, 'T13异常')
T18_model, T18_auc, T18_params = train_xgb(T18_df, 'T18异常')
T21_model, T21_auc, T21_params = train_xgb(T21_df, 'T21异常')

# 单独检验
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def stable_stratified_cv_auc(X, y, n_splits=5, n_repeats=5, random_state=42):
    """
    小样本不平衡的 Stratified CV，重复多次，计算稳定 AUC。
    
    参数:
        X: 特征 DataFrame 或 ndarray
        y: 标签 Series 或 ndarray
        n_splits: 每次 CV 的折数
        n_repeats: 重复 CV 次数
        random_state: 随机种子
    返回:
        mean_auc: 平均 AUC
        all_fold_aucs: 每折 AUC 的列表
    """
    all_fold_aucs = []

    for repeat in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state + repeat)
        fold_aucs = []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # 处理正负样本不平衡
            n_pos = sum(y_train == 1)
            n_neg = sum(y_train == 0)
            scale_pos_weight = n_neg / max(n_pos, 1)  # 防止除零

            model = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='auc',
                scale_pos_weight=scale_pos_weight,
                random_state=random_state
            )

            model.fit(X_train, y_train)

            y_pred_prob = model.predict_proba(X_test)[:,1]
            auc = roc_auc_score(y_test, y_pred_prob)
            fold_aucs.append(auc)

        all_fold_aucs.extend(fold_aucs)

    mean_auc = np.mean(all_fold_aucs)
    print(f"Stratified CV AUCs (每折): {all_fold_aucs}")
    print(f"重复 {n_repeats} 次 CV 平均 AUC: {mean_auc:.4f}")

    return mean_auc, all_fold_aucs

# ===== 使用示例 =====
# 假设 df_x 是特征, df_y 是 0/1 标签
df_x = T18_df.drop(columns=['孕妇代码','T13异常','T18异常','T21异常'], errors='ignore')
df_y = T18_df['T18异常']

mean_auc, all_aucs = stable_stratified_cv_auc(df_x, df_y, n_splits=5, n_repeats=5)

print(mean_auc)
print(all_aucs)


# 初步分别检验
from sklearn.metrics import roc_curve, auc
def get_pred_df(model, df, abnormal_col):
    X = df.drop(columns=['孕妇代码', 'T13异常', 'T18异常', 'T21异常'], errors='ignore')
    p = model.predict_proba(X)[:,1]
    return pd.DataFrame({
        '孕妇代码': df['孕妇代码'],
        f'{abnormal_col}_pred': p,
        abnormal_col: df[abnormal_col]
    })

# 得到三个模型的预测结果
pred13 = get_pred_df(T13_model, T13_df, 'T13异常')
pred18 = get_pred_df(T18_model, T18_df, 'T18异常')
pred21 = get_pred_df(T21_model, T21_df, 'T21异常')

# 合并到一个 DataFrame
all_pred = pred13.merge(pred18, on='孕妇代码').merge(pred21, on='孕妇代码')

# 真实标签：只要有一个异常就是异常
all_pred['y_true'] = all_pred[['T13异常','T18异常','T21异常']].max(axis=1)

# 预测概率：取三个模型预测概率的最大值
weights = [0.3, 0.3, 0.3]  # 举例
all_pred['y_score'] = (all_pred[['T13异常_pred','T18异常_pred','T21异常_pred']] * weights).sum(axis=1)

# ==============
# 画 ROC 曲线
# ==============
fpr, tpr, _ = roc_curve(all_pred['y_true'], all_pred['y_score'])
final_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, lw=2, label=f'Final ROC (AUC={final_auc:.4f})', color='darkorange')
plt.plot([0,1],[0,1], linestyle='--', color='navy')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("整合后用户级别 ROC")
plt.legend(loc="lower right")
plt.show()

print("最终整合 AUC:", final_auc)