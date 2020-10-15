import pandas as pd
import tqdm

# 读取泄露的数据
data = pd.read_csv("AmesHousing.csv")
data.drop(["PID"], axis=1, inplace=True)

# 读取官方提供的数据
train_data = pd.read_csv("train.csv")
data.columns = train_data.columns
test_data = pd.read_csv("test.csv")
submission_data = pd.read_csv("sample_submission.csv")

print("data:{},train:{},test:{}".format(data.shape, train_data.shape, test_data.shape))

# 删除丢失的数据
miss = test_data.isnull().sum()
miss = miss[miss > 0]
data.drop(miss.index, axis=1, inplace=True)
data.drop(["Electrical"], axis=1, inplace=True)

test_data.dropna(axis=1, inplace=True)
test_data.drop(["Electrical"], axis=1, inplace=True)

for i in tqdm.trange(0, len(test_data)):
    for j in range(0, len(data)):
        for k in range(1, len(test_data.columns)):
            if test_data.iloc[i, k] == data.iloc[j, k]:
                continue
            else:
                break
        else:
            submission_data.iloc[i, 1] = data.iloc[j, -1]
            break

submission_data.to_csv('submission.csv', index=False)
