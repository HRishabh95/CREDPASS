import pandas as pd
import numpy as np
import math

def mapfunct(x, type='exp', n=0.2):
    """
    Map 0-inf to 1-0 with some function
    Type:
    """
    if type == 'inverse':
        return 1 / (1 + n * x)
    if type == 'arctan':
        return 1 - 2 / math.pi * math.atan(n * x)
    if type == 'exp':
        return (1 / (1 + n)) ** x
    else:
        raise (NotImplementedError("Function not implemented"))

def get_weights(final_df,method=None):
    if method=='AHP':
        from pyDecision.algorithm import ahp_method
        weight_derivation = 'geometric'

        dataset = np.array([
            [1, 2],
            [1 / 2, 1]
        ])

        weights, rc = ahp_method(dataset, wd=weight_derivation)
    #
    # # Consistency Ratio
    # print('RC: ' + str(round(rc, 2)))
    # if (rc > 0.10):
    #     print('The solution is inconsistent, the pairwise comparisons must be reviewed')
    # else:
    #     print('The solution is consistent')
    elif method=='critic':
        from pyDecision.algorithm import critic_method

        docno = final_df['docno'].values
        qid = final_df['qid'].values

        fixed_dfs = final_df[['score_zscore', 'c_score']]
        fixed_dfs = fixed_dfs.to_numpy(dtype=np.float64)
        weights = np.array([0.6, 0.4])

        # Load Criterion Type: 'max' or 'min'
        criterion_type = ['max', 'max']
        weights=critic_method(fixed_dfs,criterion_type)

    else:
        weights=[0.6,0.4]
    return weights

def get_zscore(final_df):
    qid_dfs = []
    qids = np.unique(final_df.qid.values)
    for qid in qids:
        qid_df = final_df.loc[final_df['qid'] == qid]
        qid_df['score_zscore'] = (qid_df['score'] - qid_df['score'].min()) / (
                    qid_df['score'].max() - qid_df['score'].min())
        qid_dfs.append(qid_df)
    return pd.concat(qid_dfs)

def get_normalized_score(final_df):
    final_df['score_zscore']=final_df['score'].apply(mapfunct)
    return final_df


#top_10_sents_df=pd.read_csv('/home/ubuntu/rupadhyay/CREDPASS/clef2020_BM25_SRet_10_PRet_10_first_weight_fine_tunned.csv',sep='\t')
file_path='/home/ubuntu/rupadhyay/CREDPASS/trec2020_BM25_SRet_10_PRet_10_micro.csv'
#file_path='/home/ubuntu/rupadhyay/CREDPASS/clef2020_BM25_top_sentences_fine_tunned_Biobert.csv'
top_10_sents_df=pd.read_csv(file_path,sep='\t')
normalized_df = get_zscore(top_10_sents_df)
#normalized_df=get_normalized_score(top_10_sents_df)
weights = get_weights(normalized_df, method=None)
top_10_sents_df['combined_score'] = normalized_df['score_zscore'] * weights[0] + normalized_df['igs_score'] * weights[1]

qids = np.unique(top_10_sents_df.qid.values)
sorted_dfs=[]
for qid in qids:
    if qid!=28 and qid!=191001:
        qid_df=top_10_sents_df.loc[top_10_sents_df['qid']==qid]
        sorted_qid_df=qid_df.sort_values('combined_score',ascending=False).reset_index()
        sorted_qid_df['n_rank']=1
        for i in sorted_qid_df.index:
            sorted_qid_df.at[i,'n_rank']=i
        sorted_dfs.append(sorted_qid_df)

sorted_qid_df_concat=pd.concat(sorted_dfs)
sorted_qid_df_concat['Q0']='Q0'
result_df=sorted_qid_df_concat[['qid','Q0','docno','n_rank','combined_score']]
result_df.columns=['qid','Q0','docno','rank','score']
result_df.drop_duplicates(subset=['qid','docno'],inplace=True)
syntax='wa_1000_ret10'
result_df['experiment']=syntax
result_df.to_csv('/home/ubuntu/rupadhyay/CREDPASS/result/60_40_TREC_micro_%s.csv'%syntax, sep=' ', index=None, header=None)
