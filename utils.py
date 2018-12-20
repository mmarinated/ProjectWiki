import pandas as pd
import numpy as np
import subprocess

def total_time(minutes, n_gpus):
    """
    returns time in hours
    """
    return minutes / 2000 * 145000 * n_gpus / 60

def _parse_string(string):    
    if string.find(".wikipedia.org_") > 0:
        name, agent = string.split(".wikipedia.org_")
        site = "wikipedia"
    elif string.find(".mediawiki.org_") > 0:
        name, agent = string.split(".mediawiki.org_")
        site = "mediawiki"
    elif string.find(".wikimedia.org_") > 0:
        name, agent = string.split(".wikimedia.org_")
        site = "wikimedia"
    else:
        print(string)
        raise Exception()
        
    idx = name[::-1].find('_')
    language = name[-idx:]
    
    name = name[:(len(name) - idx)]
        
    idx = agent.find('_')
    access = agent[:idx]
    agent = agent[(idx+1):]
      
    return name, language, site, access, agent, string

def ParseSitesNames(df):
    info = df.Page.apply(lambda x : _parse_string(x))
    info = pd.DataFrame(list(info.values),
                    columns=["name", "lang", "site", "access", "agent", "initial"])
    return info


def ReadData(type="train_small", train="train"):
    """
    ::return:: train, test, info, index
    """
    
    assert (type in ["test", "debug1", "debug2", "debug3", "train_big", "train_big_cleaned"])
    
    PATH = "../data/google_wtts/" + type + "/"
    
    train = pd.read_csv(PATH + "train.csv", dtype="float32")
    info = pd.read_csv(PATH + "info.csv")
    
    if type == "test":
        key = pd.read_csv(PATH + "key.csv")
        # no test, sample submission instead
        test = pd.read_csv(PATH + "sample_submission.csv")
    else:
        # no need for key
        key = -1
        test = pd.read_csv(PATH + "test.csv", dtype="float32")

    index = pd.read_csv(PATH + "index.csv")
        
    return train, test, info, index
        
def _for_score(y_true, y_pred):
    np.warnings.filterwarnings('ignore')
    
    assert np.isnan(y_pred).sum().sum() == 0, "np.isnan(y_true).sum() == 0"
    
    if type(y_true) == pd.DataFrame:
        y_true.fillna(0, inplace=True)
        y_true = y_true.values.astype(np.float32)
        
    y_pred = y_pred.astype(np.float32)
    
    return y_true, y_pred
    
def SMAPE_score(y_true, y_pred, print_mean=True):
    y_true, y_pred = _for_score(y_true, y_pred)
    
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 200.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
#     return diff
    if print_mean:
        print("SMAPE: ", np.nanmean(diff))
        
    return np.nanmean(diff, axis=1)

def RMSLE_score(y_true, y_pred, print_mean=True):
    y_true, y_pred = _for_score(y_true, y_pred)
    
    assert y_true.shape == y_pred.shape
    
    score = np.sqrt(
        np.mean(
            np.power(np.log1p(y_true)-np.log1p(y_pred), 2),
            axis=1
        )
    )
        
    if print_mean:
        print("RMSLE: ", np.nanmean(score))
    
    return score


def MASE(y_true, y_pred, print_mean=True):
    """
    Computes the MEAN-ABSOLUTE SCALED ERROR forcast error for univariate time series prediction.
    
    See "Another look at measures of forecast accuracy", Rob J Hyndman
    """
    

    y_true, y_pred = _for_score(y_true, y_pred)

    errors = np.mean(
        np.abs(y_true - y_pred),
        axis=1
    )

    changes = np.mean(np.abs(np.diff(y_true) + 1), axis=1)
    
    score = errors / changes
    
    if print_mean:
        print("MASE: ", np.nanmean(score))
    
    return score


def from_log_and_correction(predictions):
    return np.round(
        np.clip(
            np.nan_to_num(
                np.expm1(predictions)
            ), 0, None
        )
    )

def SaveModel(full_name,
              y_pred, test,
              time,
              comments,
              index):
    assert index.ndim == 1
    path = "../data/google_wtts/models/"
    
    command = "mkdir " + path + full_name
    subprocess.call(command.split())
    
    # predictions
    pred_array = np.zeros((145063, 60))
    pred_array[:] = None
    pred_array[index] = y_pred
    np.save(path + full_name + '/predictions', pred_array)
    
    # score
    
    
    zeros_index = np.load("../data/google_wtts/zeros_index_for_score.npy")
    weekly_index = np.load("../data/google_wtts/weekly_index_for_score.npy")
    
    scores_for_info = []
    
    for scoring in  [SMAPE_score, RMSLE_score, MASE]:
        score = scoring(y_true=test, y_pred=y_pred, print_mean=False)
        full_score = np.zeros((145063,))
        full_score[:] = np.nan
        full_score[index] = score

        # scores = pd.read_csv(path +  "scores.csv")
        # scores[full_name] = pd.Series(full_score)
        # scores.to_csv(path +  "scores.csv", index=None)

        # fill info
        info = pd.read_csv(path + "info.csv")
        full_score_mean = np.nanmean(score) if index.shape[0] > 3000 else None

        if index.shape[0] == 2000:
            debug_score = np.nanmean(score)
        elif index.shape[0] > 10000:
            index_for_debug = pd.read_csv('../data/google_wtts/debug3/index.csv').values.reshape(-1)
            debug_score = np.nanmean(score[index_for_debug])
        else:
            print("USE either full data or debug3")
            assert False
    
        zeros_score = np.nanmean(full_score[zeros_index])
        weekly_score = np.nanmean(full_score[weekly_index])

        scores_for_info.append(full_score_mean)
        scores_for_info.append(debug_score)
        scores_for_info.append(zeros_score)
        scores_for_info.append(weekly_score)
                  
                  
                  
    scores_for_info.append(time)
    scores_for_info.append(comments)
                  
    info[full_name] = pd.Series(scores_for_info)
    info.to_csv(path + "info.csv", index=None)

    
    