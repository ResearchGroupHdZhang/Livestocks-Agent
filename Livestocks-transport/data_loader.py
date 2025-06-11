import pandas as pd
import os
import numpy as np

country_mapping = {
    'usa': '美国',
    'br': '巴西',
    'eu': '欧盟',
    'cn': '中国',
    'aus': '澳大利亚'
}
def load_datas(country, file_name="中国优化N优先v2.xlsx", province=None):
    country = country_mapping[country]
    outgoing = pd.read_excel(os.path.join(os.path.dirname(__file__), f"../data/{country}/{file_name}"), sheet_name='移出')
    incoming = pd.read_excel(os.path.join(os.path.dirname(__file__), f"../data/{country}/{file_name}"), sheet_name='移入')
    incoming.fillna(0, inplace=True)
    outgoing.fillna(0, inplace=True)

    if province:
        outgoing = outgoing.loc[outgoing['province'] == province]
        incoming = incoming.loc[incoming['province'] == province]
        
    # 去除outgoing和incoming中所有需要移动数量为0的行
    outgoing = outgoing.loc[(outgoing[[col for col in outgoing.columns if 'num' in col]].sum(axis=1) != 0)]
    incoming = incoming.loc[(incoming[[col for col in incoming.columns if 'num' in col]].sum(axis=1) != 0)]
    # outgoing = outgoing.loc[(outgoing[[col for col in outgoing.columns if 'num' in col]].sum(axis=1) != 0) & (outgoing['ammonia指标'] != 0)]
    # incoming = incoming.loc[(incoming[[col for col in incoming.columns if 'num' in col]].sum(axis=1) != 0) & (incoming['ammonia指标'] != 0)]


    # outgoing = outgoing.iloc[:10]
    # incoming = incoming.iloc[:10]

    ID_move_in = incoming[['ID', 'city', 'county']]
    ID_move_out = outgoing[['ID', 'city', 'county']]

    Amount_move_in =  incoming[[col for col in incoming.columns if 'num' in col]].astype(np.int64)
    Amount_move_out = outgoing[[col for col in outgoing.columns if 'num' in col]].astype(np.int64)
    
    N_demand_move_in = incoming[[col for col in incoming.columns if '最优氮需求' in col]]
    N_demand_Coef_move_in =  incoming[[col for col in incoming.columns if 'manure变化量' in col]]
    ammonia_move_in = incoming[[col for col in incoming.columns if 'ammonia指标' in col]]
    ammonia_Coef_move_in = incoming[[col for col in incoming.columns if '氨变化量' in col]]
    sensitivity_move_in = incoming[[col for col in incoming.columns if '敏感度' in col]]
    relative_pm25_move_in = incoming[[col for col in incoming.columns if 'PM2.5相关性' in col]]

    N_demand_move_out = outgoing[[col for col in outgoing.columns if '最优氮需求' in col]]
    N_demand_Coef_move_out = outgoing[[col for col in outgoing.columns if 'manure变化量' in col]]
    ammonia_move_out = outgoing[[col for col in outgoing.columns if 'ammonia指标' in col]]
    ammonia_Coef_move_out = outgoing[[col for col in outgoing.columns if '氨变化量' in col]]
    sensitivity_move_out = outgoing[[col for col in outgoing.columns if '敏感度' in col]]
    relative_pm25_move_out = outgoing[[col for col in outgoing.columns if 'PM2.5相关性' in col]]

    
    return (ID_move_in,
            ID_move_out,
            Amount_move_out, 
            Amount_move_in, 
            N_demand_move_in, 
            N_demand_Coef_move_in, 
            ammonia_move_in,
            ammonia_Coef_move_in,
            sensitivity_move_in, 
            relative_pm25_move_in, 
            N_demand_move_out, 
            N_demand_Coef_move_out,
            ammonia_move_out,
            ammonia_Coef_move_out,
            sensitivity_move_out,
            relative_pm25_move_out)

if __name__ == "__main__":
    load_datas("br", "巴西空间优化第一步1203.xlsx")
    print("Data loaded successfully")